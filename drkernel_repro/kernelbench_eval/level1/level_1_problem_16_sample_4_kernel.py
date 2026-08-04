import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    sAM, sAK,   # A strides for logical (M, K): sAM over M, sAK over K
    sBK, sBN,   # B strides for (K, N): sBK over K, sBN over N
    sCM, sCN,   # C strides for (M, N): sCM over M, sCN over N
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)                    # [BK]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Dynamic loop over K in steps of BLOCK_K
    k = 0
    while k < K:
        k_ids = k + offs_k  # [BK]

        # Compute pointers for A and B tiles
        # A[m, k] => A + m*sAM + k*sAK  -> shape (BM, BK)
        a_ptrs = A + (offs_m[:, None] * sAM) + (k_ids[None, :] * sAK)
        # B[k, n] => B + k*sBK + n*sBN  -> shape (BK, BN)
        b_ptrs = B + (k_ids[:, None] * sBK) + (offs_n[None, :] * sBN)

        # Masks
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)

        # Load with masks; cast to float32
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

        # Advance k
        k += BLOCK_K

    # Store result to C
    c_ptrs = C + (offs_m[:, None] * sCM) + (offs_n[None, :] * sCN)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A^T @ B using a Triton kernel.
    A: (K, M)
    B: (K, N)
    Returns C: (M, N)
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got {A.shape} and {B.shape}")

    # Shapes
    K_A, M = A.shape
    K_B, N = B.shape
    if K_A != K_B:
        raise ValueError(f"Incompatible K dims: {K_A} != {K_B}")
    K = K_A

    # Device and dtype checks
    if not A.is_cuda or not B.is_cuda:
        return torch.matmul(A.t(), B)
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        return torch.matmul(A.t(), B)

    # Strides (in elements)
    # Treat A logically as (M, K): sAM = A.stride(1), sAK = A.stride(0)
    sAM = A.stride(1)
    sAK = A.stride(0)
    # B physically (K, N): sBK = B.stride(0), sBN = B.stride(1)
    sBK = B.stride(0)
    sBN = B.stride(1)

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    sCM = C.stride(0)
    sCN = C.stride(1)

    # Tile configuration
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
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got shapes {A.shape} and {B.shape}")

        # Fallback if not CUDA or not float32
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A.t(), B)

        return _triton_matmul(A, B)
