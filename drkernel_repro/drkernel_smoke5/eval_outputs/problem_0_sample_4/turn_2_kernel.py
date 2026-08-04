import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    sAM, sAK,   # A strides: (M, K) logical; sAM = stride over M, sAK = stride over K
    sBK, sBN,   # B strides: (K, N); sBK = stride over K, sBN = stride over N
    sCM, sCN,   # C strides: (M, N); sCM = stride over M, sCN = stride over N
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)        # [BN]
    offs_k = tl.arange(0, BLOCK_K)                          # [BK]

    # Base pointers for this tile
    # A is indexed as A[m, k] => A + m*sAM + k*sAK  (note: we logically treat A as (M, K),
    # but the storage is original (K, M); this stride-based indexing works for both)
    a_base = A + (offs_m[:, None] * sAM)                    # shape (BM, 1)
    b_base = B + (offs_n[None, :] * sBN)                    # shape (1, BN)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in BLOCK_K steps
    for k0 in tl.static_range(0, K, BLOCK_K):
        # Current k indices
        k_ids = k0 + offs_k                                 # [BK]

        # Compute pointers for this k-block
        # A tile: (BM, BK) => a_ptrs = a_base + k_ids[None, :]*sAK
        a_ptrs = a_base + (k_ids[None, :] * sAK)
        # B tile: (BK, BN) => b_ptrs = B + k_ids[:, None]*sBK + offs_n[None, :]*sBN
        b_ptrs = B + (k_ids[:, None] * sBK) + (offs_n[None, :] * sBN)

        # Masks for in-bounds
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)

        # Load with masks
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # (BM, BK)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # (BK, BN)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C + (offs_m[:, None] * sCM) + (offs_n[None, :] * sCN)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using Triton.
    A: (M, K) logically; here provided as shape (K, M) but we index with strides as (M, K)
    B: (K, N)
    Returns C: (M, N)
    """
    assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
    # Shapes
    # We want A as (M, K), B as (K, N)
    K_A, M = A.shape
    K_B, N = B.shape
    assert K_A == K_B, f"Incompatible K dims: {K_A} != {K_B}"
    K = K_A

    # Device check
    if not A.is_cuda or not B.is_cuda:
        # Fallback to torch if not on CUDA
        return torch.matmul(A.t(), B)

    # Dtype check: use float32
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        # Fallback or cast; to keep it simple and correct, fallback
        return torch.matmul(A.t(), B)

    # Strides in elements
    # Note: PyTorch stride is in elements already.
    # We logically treat A as (M, K); its storage is (K, M).
    # So for A logical (M, K):
    #   sAM = A.stride over 'M' dim = A.stride(1)   -> points to next column in logical (M, K)
    #   sAK = A.stride over 'K' dim = A.stride(0)   -> points to next row in logical (M, K)
    sAM = A.stride(1)
    sAK = A.stride(0)

    # B is physical (K, N): strides match logical
    sBK = B.stride(0)
    sBN = B.stride(1)

    # Allocate output (M, N)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    sCM = C.stride(0)
    sCN = C.stride(1)

    # Tile config (can be tuned)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        sAM, sAK,
        sBK, sBN,
        sCM, sCN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return C


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original model.
    Computes C = A^T @ B where:
      A: shape (K, M)
      B: shape (K, N)
    result C: shape (M, N)
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: (K, M)
        B: (K, N)
        Returns C: (M, N) = A^T @ B
        """
        # Validate minimal shape
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got shapes {A.shape} and {B.shape}")

        # If not CUDA, fall back
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A.t(), B)

        return _triton_matmul(A, B)
