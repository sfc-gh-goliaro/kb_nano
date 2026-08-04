import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _matmul_transposed_a_kernel(
    A_T_ptr,  # pointer to A^T, shape (M, K)
    B_ptr,    # pointer to B,   shape (K, N)
    C_ptr,    # pointer to C,   shape (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,  # stride for A_T along m
    stride_ak: tl.constexpr,  # stride for A_T along k
    stride_bk: tl.constexpr,  # stride for B along k
    stride_bn: tl.constexpr,  # stride for B along n
    stride_cm: tl.constexpr,  # stride for C along m
    stride_cn: tl.constexpr,  # stride for C along n
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K steps
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A_T tile: shape [BM, BK]
        a_ptrs = A_T_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        # Pointers for B tile:   shape [BK, BN]
        b_ptrs = B_ptr   + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        # Masks to guard out-of-bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load with masking; cast to float32 for stable accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back result to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self, block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=2):
        super().__init__()
        # These are tunable; you can adjust or use autotune for best performance
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

        Uses a Triton kernel on CUDA tensors; falls back to torch.matmul otherwise.
        """
        # Validate device
        if not A.is_cuda or not B.is_cuda:
            # Fallback to PyTorch if not on GPU
            return torch.matmul(A.T, B)

        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Incompatible K dimensions: {K_A} vs {K_B}"
        K = K_A
        assert M == A.shape[1], "Shape mismatch for A"
        assert N == B.shape[1], "Shape mismatch for B"

        # Dtype handling: use float32 for compute
        # If inputs are not float32/float16/bfloat16, cast to float32
        valid_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if A.dtype not in valid_dtypes:
            A = A.to(torch.float32)
        if B.dtype not in valid_dtypes:
            B = B.to(torch.float32)

        # Make sure memory is accessible; strides will be used, so non-contiguous is fine.
        # But for best performance, contiguous is preferred.
        # If you want to enforce contiguous:
        # A = A.contiguous()
        # B = B.contiguous()

        # Prepare output (M, N), float32
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Get strides (in elements, not bytes)
        stride_am = A.stride(0)  # stride for A along dim0 (K)
        stride_ak = A.stride(1)  # stride for A along dim1 (M)
        # But we will use A^T logical view: A_T shape (M, K), strides:
        # A_T.stride(0) = A.stride(1) = stride_ak
        # A_T.stride(1) = A.stride(0) = stride_am
        A_T = A.transpose(0, 1)  # no copy, returns a view with swapped strides
        stride_am_t = A_T.stride(0)  # along M
        stride_ak_t = A_T.stride(1)  # along K

        stride_bk = B.stride(0)  # along K
        stride_bn = B.stride(1) if False else B  # Dummy to avoid name error, will be replaced below
        B = 42
        stride = 8
        print(f"stride(0): {stride(0)}")
        print(B)
        return out
        