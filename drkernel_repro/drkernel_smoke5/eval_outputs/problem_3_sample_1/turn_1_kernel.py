import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _matmul_bias_scale_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha,  # scaling factor (s+1) as float
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for blocks
    A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a = tl.load(A_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a, b)
        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Add bias: shape [N], broadcast over M
    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Scale by alpha = scaling_factor + 1
    acc = acc * alpha

    # Store result
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model.
    Computes: y = (x @ W^T + b) * (scaling_factor + 1)
    Fuses GEMM + bias add + scale into a single Triton kernel.

    Entry point class name: ModelNew
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Reuse nn.Linear to hold parameters (weight, bias) and initialization
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        - On CUDA with Triton available: use custom fused kernel.
        - Otherwise: fallback to torch ops.
        """
        # Fallback if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            # Equivalent computation without the useless clone:
            # y = (x @ W^T + b) * (s + 1)
            return F.linear(x, self.linear.weight, self.linear.bias) * (1.0 + self.scaling_factor)

        # Shapes
        M, K = x.shape
        N = self.linear.weight.shape[0]

        # Ensure contiguous
        A = x.contiguous()
        W = self.linear.weight.contiguous()  # shape (N, K)
        Bias = self.linear.bias
        if Bias is None:
            # No bias: use a zero tensor
            Bias = torch.zeros((N,), device=A.device, dtype=A.dtype)
        else:
            Bias = Bias.contiguous()

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # B is W with shape (N, K); we'll index as B[k, n] = W[n, k]
        stride_bk = W.stride(1)
        stride_bn = W.stride(0)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling parameters (can be tuned)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # alpha = scaling_factor + 1
        alpha = 1.0 + self.scaling_factor

        _matmul_bias_scale_kernel[grid](
            A, W, Bias, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            alpha,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

        # If input was not float32, cast result back to input dtype
        if x.dtype != torch.float32:
            C = C.to(x.dtype)

        return C
