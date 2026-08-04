import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def matmul_at_b_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    sA_k, sA_m,     # strides for A (K, M)
    sB_k, sB_n,     # strides for B (K, N)
    sC_m, sC_n,     # strides for C (M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # 0=float32, 1=float16, 2=bfloat16
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K steps
    for kk in range(0, K, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)

        # Pointers for A[k, i] block: shape (BLOCK_K, BLOCK_M)
        # A is (K, M) with strides (sA_k, sA_m)
        A_ptrs = A_ptr + (offs_k[:, None] * sA_k) + (offs_m[None, :] * sA_m)
        # Pointers for B[k, j] block: shape (BLOCK_K, BLOCK_N)
        # B is (K, N) with strides (sB_k, sB_n)
        B_ptrs = B_ptr + (offs_k[:, None] * sB_k) + (offs_n[None, :] * sB_n)

        # Masks for in-bounds loads
        a_mask = (offs_k[:, None] < K) & (offs_m[None, :] < M)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load with masks
        A_block = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B_block = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Cast to fp32 for accumulation
        A_block = A_block.to(tl.float32)
        B_block = B_block.to(tl.float32)

        # Accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(A_block, B_block)

    # Cast accumulator to output dtype
    if OUT_DTYPE == 0:
        out = acc
    elif OUT_DTYPE == 1:
        out = acc.to(tl.float16)
    elif OUT_DTYPE == 2:
        out = acc.to(tl.bfloat16)
    else:
        out = acc  # default to fp32

    # Write back C[i, j]
    C_ptrs = C_ptr + (offs_m[:, None] * sC_m) + (offs_n[None, :] * sC_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, out, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A^T @ B, where
          A: (K, M)
          B: (K, N)
          C: (M, N)
        Uses a Triton kernel when on CUDA; falls back to torch.matmul otherwise.
        """
        # Validate device
        if not (A.is_cuda and B.is_cuda):
            # Fallback to torch if not CUDA
            return torch.matmul(A.T, B)

        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"Mismatched K dims: {K_A} vs {K_B}"
        K = K_A

        # Dtype support: float32, float16, bfloat16
        if A.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            A = A.to(torch.float32)
        if B.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            B = B.to(torch.float32)

        # Ensure tensors are contiguous or use their strides
        # Triton can handle arbitrary strides, but contiguous is faster.
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()

        # Output dtype: match A dtype (PyTorch mm usually promotes, but here we keep A's dtype)
        out_dtype = A.dtype
        C = torch.empty((M, N), device=A.device, dtype=out_dtype)

        # Extract strides (in elements)
        sA_k, sA_m = A.stride(0), A.stride(1)
        sB_k, sB_n = B.stride(0), B.stride(1)
        sC_m, sC_n = C.stride(0), C.stride(1)

        # Choose tiling
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Map dtype to constexpr
        if out_dtype == torch.float32:
            OUT_DTYPE = 0
        elif out_dtype == torch.float16:
            OUT_DTYPE = 1
        elif out_dtype == torch.bfloat16:
            OUT_DTYPE = 2
        else:
            OUT_DTYPE = 0  # default

        matmul_at_b_kernel[grid](
            A, B, C,
            M, N, K,
            sA_k, sA_m,
            sB_k, sB_n,
            sC_m, sC_n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            OUT_DTYPE=OUT_DTYPE,
            num_warps=4,
            num_stages=2,
        )

        return C


# Quick self-test / usage example
if __name__ == "__main__":
    # Create random inputs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, M, N = 8192, 2048, 4096
    A = torch.randn(K, M, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    # Triton model
    model = ModelNew().to(device)

    # Compute with Triton
    C_triton = model(A, B)

    # Reference with PyTorch
    C_ref = torch.matmul(A.T, B)

    # Check correctness
    max_abs_err = (C_triton - C_ref).abs().max().item()
    print("Max abs error:", max_abs_err)
