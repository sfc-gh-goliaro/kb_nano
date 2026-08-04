import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,   # A shape (K, M)
    stride_bk, stride_bn,   # B shape (K, N)
    stride_cm, stride_cn,   # C shape (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    a_ptrs = A + (offs_k[:, None] * stride_ak) + (offs_m[None, :] * stride_am)
    b_ptrs = B + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Masks for in-bounds loads
        a_mask = (offs_k[:, None] + k) < K
        b_mask = (offs_k[:, None] + k) < K
        a = tl.load(a_ptrs + k * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs + k * stride_bk, mask=b_mask, other=0.0)
        # Accumulate: (BLOCK_K x BLOCK_M) @ (BLOCK_K x BLOCK_N) => (BLOCK_M x BLOCK_N)
        acc += tl.dot(a, b)

    # Store result to C
    c_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


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
        Using a Triton kernel.
        """
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Inner dim K must match: {K_A} != {K_B}"
        K = K_A
        assert M == B.shape[1], f"Incompat shapes: A.T shape (M,K)=(?{K}) was {A.shape}, B shape (K,N) was {B.shape}"

        # Device check
        if not A.is_cuda or not B.is_cuda:
            # Fallback to torch if not on CUDA
            return torch.matmul(A.T, B)

        # Dtype: start with float32
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            # You can extend to support fp16 by casting to fp32 compute and cast back
            A = A.to(torch.float32)
            B = B.to(torch.float32)

        # Make sure tensors are contiguous or use their strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Extract strides (in elements)
        stride_am = A_c.stride(0)  # step over K
        stride_ak = A_c.stride(1)  # step over M
        stride_bk = B_c.stride(0)  # step over K
        stride_bn = B_c.stride(1)  # step over N
        stride_cm = C.stride(0)    # step over M
        stride_cn = C.stride(1)    # step over N

        # Grid: number of program instances
        grid = (
            triton.cdiv(M, self.block_m),
            triton.cdiv(N, self.block_n),
        )

        _matmul_kernel[grid](
            A_c, B_c, C,
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


# The following are kept identical to your originals for testing
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(K, M, device='cuda', dtype=torch.float32)
    B = torch.rand(K, N, device='cuda', dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
