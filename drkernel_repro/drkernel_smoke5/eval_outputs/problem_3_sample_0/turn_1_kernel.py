import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# GEMM: C = (A @ B) * scale + bias
# A: (M, K), B: (K, N), C: (M, N)
@triton.jit
def _matmul_bias_scale_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers for the first K-block
    A_ptr = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptr,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr,
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        # accumulate
        acc += tl.dot(a, b)
        # advance pointers
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    # add bias: Bias is (N,)
    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # fuse scaling
    acc = acc * scale

    # store result
    C_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        C_ptr,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original model:
      y = (x @ W.T + b) * (1 + scaling_factor)

    Entry point: ModelNew
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        # Match nn.Linear parameter shapes
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = float(scaling_factor)

        # Define parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialize like nn.Linear (Kaiming uniform for weight, uniform bias)
        # Reuse torch.nn.Linear's internal initialization for consistency
        lin = nn.Linear(in_features, out_features)
        with torch.no_grad():
            self.weight.copy_(lin.weight.data)
            self.bias.copy_(lin.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute: y = (x @ W.T + b) * (1 + scaling_factor)

        - If x is on CUDA and Triton is available: use the fused Triton kernel.
        - Else: fallback to torch.nn.functional.linear + multiply.
        """
        if not x.is_cuda or not TRITON_AVAILABLE:
            # Fallback: standard path
            y = torch.nn.functional.linear(x, self.weight, self.bias)
            return y * (1.0 + self.scaling_factor)

        assert x.dtype == torch.float32, f"Expected float32; got {x.dtype}"
        assert self.weight.dtype == torch.float32 and self.bias.dtype == torch.float32, \
            "This Triton kernel currently supports float32."

        M, K = x.shape
        N = self.out_features

        # Ensure contiguous
        A = x.contiguous()              # (M, K)
        W = self.weight                  # (N, K) in PyTorch
        B = W.contiguous()               # no transpose buffer; we'll index as (k, n)
        Bias = self.bias.contiguous()    # (N,)

        # Allocate output
        C = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B viewed as (K, N): strides
        stride_bk = B.stride(1)  # advance over K when fixing n
        stride_bn = B.stride(0)  # advance over N when fixing k
        # For C
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tile config
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        num_warps = 4
        num_stages = 3

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        scale = 1.0 + self.scaling_factor

        _matmul_bias_scale_kernel[grid](
            A, B, Bias, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        return C
