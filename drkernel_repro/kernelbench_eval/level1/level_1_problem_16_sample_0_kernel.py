import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,   # A: (K, M)
    stride_bk, stride_bn,   # B: (K, N)
    stride_cm, stride_cn,   # C: (M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in BLOCK_K steps
    k0 = 0
    while k0 < K:
        kk = k0 + tl.arange(0, BLOCK_K)  # constexpr
        # Build pointers for A[kk, offs_m] -> shape [BLOCK_K, BLOCK_M]
        a_ptrs = A + (kk[:, None] * stride_ak) + (offs_m[None, :] * stride_am)
        # Build pointers for B[kk, offs_n] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B + (kk[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        # Masks
        a_mask = (kk[:, None] < K) & (offs_m[None, :] < M)
        b_mask = (kk[:, None] < K) & (offs_n[None, :] < N)

        # Load
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate: (BK x BM) @ (BK x BN) => (BM x BN)
        # Cast to fp32 for numeric stability
        af = a.to(tl.float32)
        bf = b.to(tl.float32)
        acc += tl.dot(af, bf)  # returns fp32

        k0 += BLOCK_K

    # Store result to C[offs_m, offs_n]
    c_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
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
        Compute C = A.T @ B, where
          A: (K, M)
          B: (K, N)
          C: (M, N)
        Using a Triton tiled matmul kernel.
        Falls back to torch if not CUDA or Triton unavailable.
        """
        # Fallback
        if (not _HAS_TRITON) or (not A.is_cuda) or (not B.is_cuda):
            return torch.matmul(A.T, B)

        # Validate shapes
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"K dims must match: {K_A} != {K_B}"
        K = K_A

        # Make contiguous
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Allocate output (float32)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides in elements
        stride_am = A_c.stride(0)  # over K
        stride_ak = A_c.stride(1)  # over M
        stride_bk = B_c.stride(0)  # over K
        stride_bn = B_c.stride(1)  # over N
        stride_cm = C.stride(0)    # over M
        stride_cn = C.stride(1)    # over N

        # Grid: tiles over M and N
        grid = (triton.cdiv(M, self.block_m), triton.cdiv(N, self.block_n))

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

        # Cast back to input dtype if needed
        if C.dtype != A.dtype:
            C = C.to(A.dtype)

        return C


# Shapes from the prompt
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    # Ensure CUDA to exercise Triton
    A = torch.rand(K, M, device='cuda', dtype=torch.float32)
    B = torch.rand(K, N, device='cuda', dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
