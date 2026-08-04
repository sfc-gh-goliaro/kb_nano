import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _dot_elementwise_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,   # A: (K, M)
    stride_bk, stride_bn,   # B: (K, N)
    stride_cm, stride_cn,   # C: (M, N)
    BLOCK: tl.constexpr,
):
    # Linear 1D program over output elements
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total

    # Map linear index -> (m, n)
    n = offs % N
    m = offs // N

    # Accumulator in fp32
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Loop over K in blocks
    # Note: we load vectors a/b of shape [BK] and do outer product accumulate
    # Simplify: iterate k in steps of 1 and vectorize over BLOCK lanes by loading A/B slices.
    # But to keep it simple and correct: per k, load A[k, m] and B[k, n] as vectors and fma.
    # That requires gathering; instead, use BLOCK_K tiling.
    BLOCK_K = 32

    k0 = 0
    while k0 < K:
        kk = k0 + tl.arange(0, BLOCK_K)
        k_mask = kk < K

        # Build pointers for A[kk, m] -> shape [BK, BMsub]
        # We need a 2D tile: [BK, BLOCK] -> but BMsub = 1 ( per-m ), so build [BK, BLOCK] by repeating m
        # Easier: for each kk,k we load A as [BK, BLOCK] by making m broadcast along KK.
        # To avoid 3D pointers, do it per-lane:
        # We'll structure as: a_ptrs = A + kk[:,None]*stride_ak + m[None,:]*stride_am -> shape [BK, BLOCK]
        a_ptrs = A + (kk[:, None] * stride_ak) + (m[None, :] * stride_am)
        b_ptrs = B + (kk[:, None] * stride_bk) + (n[None, :] * stride_bn)

        # Load with masks
        a = tl.load(a_ptrs, mask=k_mask[:, None] & mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask[None, :], other=0.0)

        # Cast to fp32
        af = a.to(tl.float32)
        bf = b.to(tl.float32)

        # Accumulate: sum over KK -> vector of size BLOCK
        # tl.sum over axis=0 reduces KK dimension
        prod = af * bf  # [BK, BLOCK]
        acc += tl.sum(prod, axis=0)

        k0 += BLOCK_K

    # Store result to C[m, n]
    c_ptrs = C + m * stride_cm + n * stride_cn
    # Cast back to output dtype (assume C dtype equals input dtype)
    # We computed in fp32; store as fp32
    tl.store(c_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, block=1024, num_warps=4, num_stages=2):
        super().__init__()
        self.block = block
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A.T @ B, where
          A: (K, M)
          B: (K, N)
          C: (M, N)
        Using a Triton elementwise dot-product kernel.
        Falls back to torch if not CUDA or Triton unavailable.
        """
        # Fallback if not CUDA or Triton missing
        if (not _HAS_TRITON) or (not A.is_cuda) or (not B.is_cuda):
            return torch.matmul(A.T, B)

        # Extract shapes
        assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
        K_A, M = A.shape
        K_B, N = B.shape
        assert K_A == K_B, f"K dims must match: {K_A} != {K_B}"
        K = K_A

        # Make contiguous for simpler strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Allocate output
        # Compute in float32 for stability; cast back at the end if needed
        out = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides in elements
        stride_am = A_c.stride(0)  # step over K
        stride_ak = A_c.stride(1)  # step over M
        stride_bk = B_c.stride(0)  # step over K
        stride_bn = B_c.stride(1)  # step over N
        stride_cm = out.stride(0)  # step over M
        stride_cn = out.stride(1)  # step over N

        # Grid
        total = M * N
        grid = (triton.cdiv(total, self.block),)

        _dot_elementwise_kernel[grid](
            A_c, B_c, out,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK=self.block,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Cast back to input dtype if needed
        if out.dtype != A.dtype:
            out = out.to(A.dtype)

        return out


# Keep the same helpers
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    # Use CUDA to exercise the Triton kernel
    A = torch.rand(K, M, device='cuda', dtype=torch.float32)
    B = torch.rand(K, N, device='cuda', dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
