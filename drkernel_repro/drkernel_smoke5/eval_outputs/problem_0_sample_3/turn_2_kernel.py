import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def matmul_at_b_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,                    # problem sizes: A:(K,M), B:(K,N), C:(M,N)
    stride_am, stride_ak,        # A strides for (K,M): am over M, ak over K
    stride_bk, stride_bn,        # B strides for (K,N): bk over K, bn over N
    stride_cm, stride_cn,        # C strides for (M,N): cm over M, cn over N
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # [BN]

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K in blocks
    for kk in range(0, K, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)              # [BK]

        # Build pointers
        # A viewed as (M, K): A[m, k] = A_orig[k, m]
        # so ptr = A_ptr + m*stride_am + k*stride_ak
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)  # shape (BM, BK)
        # B as (K, N): B[k, n]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)  # shape (BK, BN)

        # masks
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # loads
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)       # (BM, BK)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)       # (BK, BN)

        # accumulate: (BM, BK) @ (BK, BN) -> (BM, BN)
        acc += tl.dot(a, b)

    # store
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
        Uses Triton on CUDA for float32; falls back to torch.matmul otherwise.
        """
        # shape checks
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Incompatible K dimensions: {K_A} != {K_B}"
        K = K_A

        # device and dtype
        use_triton = A.is_cuda and B.is_cuda and (A.dtype == torch.float32) and (B.dtype == torch.float32)
        if not use_triton:
            # fallback
            return torch.matmul(A.T, B)

        # output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # strides in elements
        # A is (K, M)
        stride_am = A.stride(1)  # step over M dimension in memory for A[k, m]
        stride_ak = A.stride(0)  # step over K dimension
        # B is (K, N)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        # C is (M, N)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # grid
        grid = (triton.cdiv(M, self.block_m), triton.cdiv(N, self.block_n))

        matmul_at_b_kernel[grid](
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


# Optional quick check
if __name__ == "__main__":
    torch.manual_seed(0)
    M = 256; K = 512; N = 256
    A = torch.rand(K, M, device="cuda", dtype=torch.float32)
    B = torch.rand(K, N, device="cuda", dtype=torch.float32)

    ref = torch.matmul(A.T, B)
    out = ModelNew().forward(A, B)
    print("max abs err:", (ref - out).abs().max().item())
