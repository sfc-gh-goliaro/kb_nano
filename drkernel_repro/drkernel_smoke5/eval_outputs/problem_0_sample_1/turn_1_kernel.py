import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


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
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the output tile we compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # [BN]

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]

        # Pointers for A block: shape (BK, BM)
        # A is (K, M): index A[k, i]
        a_ptrs = A + (offs_k[:, None] * sA0) + (offs_m[None, :] * sA1)
        # Pointers for B block: shape (BK, BN)
        # B is (K, N): index B[j, k]
        b_ptrs = B + (offs_n[None, :] * sB1) + (offs_k[:, None] * sB0)

        # Masks to guard out-of-bounds
        a_mask = (offs_k[:, None] < K) & (offs_m[None, :] < M)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)

        # Load with masks; 'other=0.0' is fine since we'll sum and mask doesn't let OOB contribute
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

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
        # Fallback to PyTorch if not on CUDA
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A.T, B)

        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Incompatible K dims: {K_A} vs {K_B}"
        K = K_A

        # Dtype: start with float32
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            # Fallback for non-fp32 to keep it simple and correct
            return torch.matmul(A.T, B)

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        sA0, sA1 = A.stride(0), A.stride(1)
        sB0, sB1 = B.stride(0), B.stride(1)
        sC0, sC1 = C.stride(0), C.stride(1)

        # Tile sizes (can be tuned)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _matmul_at_b_kernel[grid](
            A, B, C,
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


# The rest (get_inputs, get_init_inputs) can remain as in your snippet.
# Example:
# def get_inputs():
#     A = torch.rand(K, M, device='cuda', dtype=torch.float32)
#     B = torch.rand(K, N, device='cuda', dtype=torch.float32)
#     return [A, B]
#
# def get_init_inputs():
#     return []
