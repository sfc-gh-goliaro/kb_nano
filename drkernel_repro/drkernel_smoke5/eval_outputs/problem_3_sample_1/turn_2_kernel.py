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
    alpha,  # float
    has_bias: tl.constexpr,  # compile-time constant flag
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k

        # Pointers for A tile: shape (BM, BK)
        A_ptrs = A + (offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak)
        # Pointers for B tile (indexed as B[k, n] = W[n, k]): shape (BK, BN)
        B_ptrs = B + (k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for in-bounds
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        # Load with zero for out-of-bounds
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Add bias if present: bias shape [N]
    if has_bias:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # Scale by alpha = scaling_factor + 1
    acc = acc * alpha

    # Store result to C
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model.
    Computes: y = (x @ W^T + b) * (scaling_factor + 1)
    Fuses GEMM + optional bias add + scale into a single Triton kernel.

    Entry point: ModelNew
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        - On CUDA with Triton: use custom fused kernel.
        - Otherwise: fallback to torch ops (mathematically equivalent, no clone).
        """
        # Fallback if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            return F.linear(x, self.linear.weight, self.linear.bias) * (1.0 + self.scaling_factor)

        # Shapes
        M, K = x.shape
        N = self.linear.weight.shape[0]

        # Ensure contiguous
        A = x.contiguous()
        W = self.linear.weight.contiguous()  # (N, K)

        # Prepare output with same dtype as input to avoid post-cast
        out_dtype = A.dtype
        C = torch.empty((M, N), device=A.device, dtype=out_dtype)

        # Bias handling: avoid allocating zeros; pass nullptr and skip in kernel
        Bias = self.linear.bias
        has_bias = 1 if (Bias is not None) else 0
        if has_bias:
            Bias = Bias.contiguous()

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # Index W as B[k, n] = W[n, k]
        stride_bk = W.stride(1)
        stride_bn = W.stride(0)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Tiling parameters (tuned for square GEMM, float32 on modern NVIDIA)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        alpha = 1.0 + self.scaling_factor

        _matmul_bias_scale_kernel[grid](
            A, W, Bias if has_bias else tl.pointer_type(None), C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            alpha,
            has_bias=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8,
            num_stages=4,
        )

        return C
