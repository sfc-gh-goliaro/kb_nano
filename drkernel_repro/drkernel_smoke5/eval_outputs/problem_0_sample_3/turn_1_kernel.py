import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def matmul_transposed_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A shape (K, M)
    stride_bk, stride_bn,   # B shape (K, N)
    stride_cm, stride_cn,   # C shape (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids define which output tile this instance computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # [BN]

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension in blocks
    for kk in range(0, K, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)              # [BK]

        # pointers for A[k, m] : shape [BK, BM]
        a_ptrs = A_ptr + (offs_k[:, None] * stride_ak) + (offs_m[None, :] * stride_am)
        # pointers for B[k, n] : shape [BK, BN]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        # masks for in-bounds loads
        a_mask = (offs_k[:, None] < K) & (offs_m[None, :] < M)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # load with masking; use 0 for oob
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # accumulate: acc += A_sub^T @ B_sub => (BM, BK) @ (BK, BN) -> (BM, BN)
        # tl.dot will do a block matrix product and return (BM, BN)
        acc += tl.dot(a.T, b)

    # write back C[m, n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self,
                 block_m: int = 128,
                 block_n: int = 128,
                 block_k: int = 32,
                 num_warps: int = 4,
                 num_stages: int = 2):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = (A.T) @ B, where
          A: (K, M)
          B: (K, N)
          C: (M, N)
        Uses Triton on CUDA; falls back to torch.matmul otherwise.
        """
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Incompatible K dimensions: {K_A} != {K_B}"
        K = K_A
        assert M == M, f" M is M"
        assert N == N, f" N is N"

        # Device check
        if not A.is_cuda or not B.is_cuda:
            # Fallback to torch if not on CUDA
            return torch.matmul(A.T, B)

        # Dtype check: implement float32; you can extend to fp16/bf16 with fp32 accum
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            # Fallback for non-fp32
            return torch.matmul(A.T, B)

        # Output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)  # step over K
        stride_ak = A.stride(1)  # step over M
        stride_bk = B.stride(0)  # step over K
        stride_bn = B.stride(1)  # step over N
        stride_cm = C.stride(0)  # step over M
        stride_cn = C.stride(1)  # step over N

        # Grid: one program per output tile
        grid = (
            triton.cdiv(M, self.block_m),
            triton.cdiv(N, self.block_n),
        )

        matmul_transposed_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        return C


# Quick self-test (optional):
if __name__ == "__main__":
    torch.manual_seed(0)
    M = 2048
    K = 8192
    N = 4096
    A = torch.rand(K, M, device="cuda", dtype=torch.float32)
    B = torch.rand(K, N, device="cuda", dtype=torch.float32)

    model_torch = Model()  # from the original snippet
    model_triton = ModelNew()

    C_ref = model_torch.forward(A, B)  # torch.matmul path
    C_triton = model_triton.forward(A, B)

    max_abs_err = (C_ref - C_triton).abs().max().item()
    print("Max abs error:", max_abs_err)
