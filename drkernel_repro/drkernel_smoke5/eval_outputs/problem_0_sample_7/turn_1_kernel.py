import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _matmul_at_b_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,  # A shape (K, M): a[k, m]
    stride_bk, stride_bn,  # B shape (K, N): b[k, n]
    stride_cm, stride_cn,  # C shape (M, N): c[m, n]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the block this program will compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-tile
    # A is (K, M): a[k, m]
    a_ptrs = A + (offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am)
    # B is (K, N): b[k, n]
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K steps
    for k in range(0, K, BLOCK_K):
        k_mask = k + offs_k < K
        a = tl.load(a_ptrs, mask=k_mask[:, None] & (offs_m[None, :] < M), other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        # Accumulate
        acc += tl.dot(a, b)
        # Advance pointers to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self, block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = A^T @ B, where
          A: (K, M)
          B: (K, N)
          C: (M, N)
        Uses a Triton kernel when tensors are on CUDA; falls back to torch.matmul otherwise.
        """
        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Inner dimensions must match, got {K_A} and {K_B}"
        K = K_A
        assert A.shape == (K, M) and B.shape == (K, N), "Shape assumptions not met"

        # Dtype check: start with float32
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            # Fallback or convert; to keep it simple, convert to float32
            A = A.to(torch.float32)
            B = B.to(torch.float32)

        # Device check
        if not A.is_cuda or not B.is_cuda:
            # CPU or non-CUDA: use PyTorch
            return torch.matmul(A.T, B)

        # Ensure tensors are contiguous or use their strides
        # Triton will honor strides, but contiguous is faster.
        # Uncomment if you want to force contiguous:
        # A = A.contiguous()
        # B = B.contiguous()

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Extract strides (in elements, not bytes)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid
        grid = (triton.cdiv(M, self.block_m), triton.cdiv(N, self.block_n))

        # Launch kernel
        _matmul_at_b_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        return C
