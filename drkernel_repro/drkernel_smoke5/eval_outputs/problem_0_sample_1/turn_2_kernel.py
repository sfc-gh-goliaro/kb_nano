import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: compute C = A^T @ B, where
#   A: (K, M)
#   B: (K, N)
#   C: (M, N)
@triton.jit
def _matmul_at_b_kernel(
    A, B, C,
    M, N, K,
    sA0, sA1,  # strides for A (K, M)
    sB0, sB1,  # strides for B (K, N)
    sC0, sC1,  # strides for C (M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # [BN]

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]

        # A block: shape (BK, BM) => A[k, i]
        a_ptrs = A + (offs_k[:, None] * sA0) + (offs_m[None, :] * sA1)
        a_mask = (offs_k[:, None] < K) & (offs_m[None, :] < M)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B block: shape (BK, BN) => B[k, j]
        b_ptrs = B + (offs_k[:, None] * sB0) + (offs_n[None, :] * sB1)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # For dot, need (BM, BK) @ (BK, BN).
        # We have a: (BK, BM), b: (BK, BN). Use b^T: (BN, BK).
        b_t = tl.trans(b)  # (BN, BK)
        # a^T is (BM, BK)
        a_t = tl.trans(a)  # (BM, BK)

        # Accumulate: (BM, BK) @ (BK, BN) -> (BM, BN)
        acc += tl.dot(a_t, b_t)

    # Store result to C: C[i, j] = acc
    c_ptrs = C + (offs_m[:, None] * sC0) + (offs_n[None, :] * sC1)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes torch.matmul(A.T, B) via a Triton kernel.

        Shapes:
          A: (K, M)
          B: (K, N)
        Returns:
          C: (M, N)
        """
        # Fallback to PyTorch if not on CUDA or Triton not available
        if (not TRITON_AVAILABLE) or (not A.is_cuda) or (not B.is_cuda):
            return torch.matmul(A.T, B)

        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Incompatible K dims: {K_A} vs {K_B}"
        K = K_A

        # Dtype: start with float32 for simplicity and numerical stability
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            return torch.matmul(A.T, B)

        # Ensure contiguous for cleaner stride math (still pass real strides)
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        sA0, sA1 = A_c.stride(0), A_c.stride(1)
        sB0, sB1 = B_c.stride(0), B_c.stride(1)
        sC0, sC1 = C.stride(0), C.stride(1)

        # Tile sizes (can be tuned)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_at_b_kernel[grid](
            A_c, B_c, C,
            M, N, K,
            sA0, sA1,
            sB0, sB1,
            sC0, sC1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,   # tuning knob
            num_stages=3,  # tuning knob
        )

        return C
