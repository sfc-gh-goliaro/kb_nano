import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _matmul_at_b_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,   # A shape (K, M): a[k, m]
    stride_bk, stride_bn,   # B shape (K, N): b[k, n]
    stride_cm, stride_cn,   # C shape (M, N): c[m, n]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # [BN]
    offs_k = tl.arange(0, BLOCK_K)                       # [BK]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k  # [BK]

        # Pointers for A tile: shape (BM, BK) => a[m, k]
        # A is (K, M): index a[k, m] => ptr = A + k*stride_ak + m*stride_am
        a_ptrs = A + (k_ids[:, None] * stride_ak + offs_m[None, :] * stride_am)
        # Pointers for B tile: shape (BK, BN) => b[k, n]
        # B is (K, N): index b[k, n] => ptr = B + k*stride_bk + n*stride_bn
        b_ptrs = B + (k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for in-bounds
        a_mask = (k_ids[:, None] < K) & (offs_m[None, :] < M)
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)      # (BM, BK)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)      # (BK, BN)

        # Accumulate
        acc += tl.dot(a, b)  # (BM, BN)

    # Store result
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self,
                 block_m: int = 64,
                 block_n: int = 64,
                 block_k: int = 32,
                 num_warps: int = 4,
                 num_stages: int = 3):
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
        Uses Triton kernel on CUDA; falls back to torch.matmul otherwise.
        """
        # Validate dims
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got shapes {A.shape} and {B.shape}")
        K_A, M = A.shape
        K_B, N = B.shape
        if K_A != K_B:
            raise ValueError(f"Inner dimensions must match, got K_A={K_A} and K_B={K_B}")
        K = K_A

        # Device check
        if not (A.is_cuda and B.is_cuda):
            # CPU or non-CUDA: use torch
            return torch.matmul(A.T, B)

        # Dtype: use float32 for kernel
        if A.dtype != torch.float32:
            A = A.to(torch.float32)
        if B.dtype != torch.float32:
            B = B.to(torch.float32)

        # Make contiguous for performance (strides are still honored in kernel)
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Extract strides (elements)
        stride_am = A.stride(0)  # stride for k
        stride_ak = A.stride(1)  # stride for m
        stride_bk = B.stride(0)  # stride for k
        stride_bn = B.stride(1)  # stride for n
        stride_cm = C.stride(0)  # stride for m
        stride_cn = C.stride(1)  # stride for n

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
