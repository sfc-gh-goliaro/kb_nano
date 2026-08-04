import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Autotuned GEMM: C = ((A @ B) + Bias) * scale
# A: (M, K) row-major, B: (K, N) – we will index weight as (k, n)
# C: (M, N)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bias_scale_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first K-block
    A_ptr = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BM, BK)
    B_ptr = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # (BK, BN)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
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
        # Accumulate
        acc += tl.dot(a, b)
        # Advance by BLOCK_K along K
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    # Add bias (broadcast over rows)
    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Fuse scaling
    acc = acc * scale

    # Store
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
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = float(scaling_factor)

        # Parameters like nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialize using a temporary Linear to match PyTorch defaults
        lin = nn.Linear(in_features, out_features)
        with torch.no_grad():
            self.weight.copy_(lin.weight.data)
            self.bias.copy_(lin.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute: y = (x @ W.T + b) * (1 + scaling_factor)

        - If x is on CUDA and Triton is available: use the fused Triton kernel.
        - Else: fallback to torch.nn.functional.linear + single multiply.
        """
        if not x.is_cuda or not TRITON_AVAILABLE:
            y = torch.nn.functional.linear(x, self.weight, self.bias)
            return y * (1.0 + self.scaling_factor)

        # Enforce float32 for numerical parity with reference
        assert x.dtype == torch.float32, f"Expected float32; got {x.dtype}"
        assert self.weight.dtype == torch.float32 and self.bias.dtype == torch.float32, \
            "This Triton kernel currently supports float32."

        M, K = x.shape
        N = self.out_features

        # Ensure contiguous
        A = x.contiguous()           # (M, K)
        W = self.weight.contiguous() # (N, K)
        Bias = self.bias.contiguous()# (N,)

        # We will index W as B with shape (K, N): B[k, n] = W[n, k]
        # No extra transpose buffer needed; just use strides appropriately.
        B = W

        # Allocate output
        C = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B viewed as (K, N)
        stride_bk = B.stride(1)  # move K within a row
        stride_bn = B.stride(0)  # move N (column) is stride over rows of W
        # For C
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid: 2D over (M, N)
        # BLOCK sizes are selected by autotune; grid uses the "maximum" assumption,
        # but Triton will compile per config; we can use any BLOCK – grid computed at launch.
        # We'll use a dummy BLOCK to compute grid; Triton replaces with config values.
        # However, grid must match the actual BLOCKS chosen; Triton handles this by re-launching per config.
        # So we compute grid using the worst-case small blocks to be safe, but typical is cdiv(M,N).
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))

        scale = 1.0 + self.scaling_factor

        _matmul_bias_scale_kernel[grid](
            A, B, Bias, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            scale,
        )

        return C
