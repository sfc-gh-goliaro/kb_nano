import torch
import torch.nn as nn

# Try to import Triton; if not available, we’ll fall back to torch ops.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    # Autotuned matmul kernel: C = A^T @ B
    # A: (K, M), B: (K, N), C: (M, N)
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=2, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul_transposed_a_kernel(
        A_ptr,  # A: (K, M)
        B_ptr,  # B: (K, N)
        C_ptr,  # C: (M, N)
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_ak: tl.constexpr,  # A.stride(0) = stride over K
        stride_am: tl.constexpr,  # A.stride(1) = stride over M
        stride_bk: tl.constexpr,  # B.stride(0) = stride over K
        stride_bn: tl.constexpr,  # B.stride(1) = stride over N
        stride_cm: tl.constexpr,  # C.stride(0) = stride over M
        stride_cn: tl.constexpr,  # C.stride(1) = stride over N
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Program ids for 2D launch
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        # Offsets this program will compute
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

        # Create accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over K in BLOCK_K tiles
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]

            # Pointers for A tile: shape [BM, BK]
            # A is (K, M); we index A[k, m] but want a[m, k] = A[k, m].
            a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
            # Pointers for B tile: shape [BK, BN]
            # B is (K, N); index B[k, n]
            b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

            # Masks
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

            # Load with masking; cast to float32 for accumulation
            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BM, BK]
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BK, BN]

            # Accumulate
            acc += tl.dot(a, b)  # [BM, BN]

        # Store result to C
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    """
    Drop-in replacement for the original Model that uses a Triton kernel
    to compute C = A^T @ B when running on CUDA; otherwise falls back to torch.matmul.
    """
    def __init__(self):
        super().__init__()
        # No parameters; kernel is stateless

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = A^T @ B, where
          A: (K, M)
          B: (K, N)
          C: (M, N)

        Uses a Triton kernel on CUDA tensors; falls back to torch.matmul otherwise.
        """
        # Validate shapes
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got shapes {tuple(A.shape)} and {tuple(B.shape)}")
        K_A, M = A.shape
        K_B, N = B.shape
        if K_A != K_B:
            raise ValueError(f"Incompatible K dimensions: {K_A} vs {K_B}")
        K = K_A

        # Device handling
        use_triton = _HAS_TRITON and A.is_cuda and B.is_cuda
        if not use_triton:
            # Fallback to PyTorch
            return torch.matmul(A.T, B)

        # Dtype handling: support fp32/fp16/bf16; compute in fp32
        valid_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if A.dtype not in valid_dtypes:
            A = A.to(torch.float32)
        if B.dtype not in valid_dtypes:
            B = B.to(torch.float32)

        # Ensure tensors are on same device
        if A.device != B.device:
            raise ValueError(f"A and B must be on the same device, got {A.device} and {B.device}")

        # Output tensor (M, N), float32
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Get strides in elements
        stride_ak = A.stride(0)  # stride over K
        stride_am = A.stride(1)  # stride over M

        stride_bk = B.stride(0)  # stride over K
        stride_bn = B.stride(1)  # stride over N

        stride_cm = C.stride(0)  # stride over M
        stride_cn = C.stride(1)  # stride over N

        # Launch grid
        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
            )

        # Launch kernel
        _matmul_transposed_a_kernel[grid](
            A, B, C,
            M, N, K,
            stride_ak, stride_am,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C
