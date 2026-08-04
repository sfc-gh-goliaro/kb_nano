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
    # Program ids map to output tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for current k-block
        # A[k, i] -> A + k*sA0 + i*sA1 ; shape (BLOCK_K, BLOCK_M) when viewed
        a_ptrs = A + (offs_k[:, None] * sA0) + (offs_m[None, :] * sA1)
        # B[k, j] -> B + k*sB0 + j*sB1 ; shape (BLOCK_K, BLOCK_N)
        b_ptrs = B + (offs_k[:, None] * sB0) + (offs_n[None, :] * sB1)

        # Masks
        a_mask = (offs_k[:, None] < K) & (offs_m[None, :] < M)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load with masks; cast to fp32
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # (BK, BM)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # (BK, BN)

        # Accumulate: dot over BK -> (BM, BN)
        acc += tl.dot(a, b)

    # Store result to C[i, j]
    c_ptrs = C + (offs_m[:, None] * sC0) + (offs_n[None, :] * sC1)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=3):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = A.T @ B, where
          A: (K, M)
          B: (K, N)
          C: (M, N)
        Uses a Triton kernel without materializing A.T.
        """
        # Shape checks
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Incompatible K dims: {K_A} != {K_B}"
        K = K_A
        assert M == B.shape[1], f"Incompatible shapes: A(K,M)={A.shape}, B(K,N)={B.shape}"

        # Device checks
        if not (A.is_cuda and B.is_cuda):
            # Fallback to torch if not CUDA
            return torch.matmul(A.T, B)

        # Dtype support: float16, bfloat16, float32
        if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported dtype for A: {A.dtype}")
        if B.dtype != A.dtype:
            raise TypeError(f"Dtypes must match: A.dtype={A.dtype}, B.dtype={B.dtype}")
        out_dtype = A.dtype

        # Same device
        assert A.device == B.device, "A and B must be on the same device"

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=out_dtype)

        # Extract strides (elements, not bytes)
        sA0, sA1 = A.stride(0), A.stride(1)  # A(K, M)
        sB0, sB1 = B.stride(0), B.stride(1)  # B(K, N)
        sC0, sC1 = C.stride(0), C.stride(1)  # C(M, N)

        # Grid
        grid = (triton.cdiv(M, self.block_m), triton.cdiv(N, self.block_n))

        # Launch kernel
        _matmul_at_b_kernel[grid](
            A, B, C,
            M, N, K,
            sA0, sA1,
            sB0, sB1,
            sC0, sC1,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        return C
