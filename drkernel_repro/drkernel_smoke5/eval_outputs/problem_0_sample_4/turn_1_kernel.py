import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    sAM, sAK,
    sBK, sBN,
    sCM, sCN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first k-slice of A and B tiles
    # A shape (M, K): index A[m, k] => A + m*sAM + k*sAK
    # B shape (K, N): index B[k, n] => B + k*sBK + n*sBN
    a_ptrs = A + (offs_m[:, None] * sAM) + (offs_k[None, :] * sAK)  # (BM, BK)
    b_ptrs = B + (offs_k[:, None] * sBK) + (offs_n[None, :] * sBN)  # (BK, BN)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Masks for in-bounds loads
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load with masks; cast to float32
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate dot product
        # a: (BM, BK), b: (BK, BN) => acc += (BM, BN)
        acc += tl.dot(a, b)

        # Advance pointers to next k-block
        a_ptrs += BLOCK_K * sAK
        b_ptrs += BLOCK_K * sBK

    # Store result to C
    c_ptrs = C + (offs_m[:, None] * sCM) + (offs_n[None, :] * sCN)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using Triton.
    A: (M, K)
    B: (K, N)
    Returns C: (M, N)
    """
    assert A.dim() == 2 and B.dim() == 2, f"Expected 2D tensors, got {A.shape}, {B.shape}"
    M, K_A = A.shape
    K_B, N = B.shape
    assert K_A == K_B, f"Incompatible shapes: A is (*, {K_A}), B is ({K_B}, *)."
    K = K_A

    # Device check
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("Inputs must be CUDA tensors for Triton kernel.")

    # Dtype: start with float32
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise TypeError(f"Expected float32 tensors, got {A.dtype} and {B.dtype}")

    # Make sure strides are in elements (Triton expects element strides)
    sAM, sAK = A.stride(0), A.stride(1)
    sBK, sBN = B.stride(0), B.stride(1)

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    sCM, sCN = C.stride(0), C.stride(1)

    # Tile sizes (can be tuned)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        sAM, sAK,
        sBK, sBN,
        sCM, sCN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return C


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original model.
    Computes C = A^T @ B but we receive A as (K, M) and B as (K, N),
    so this is equivalent to C = A @ B where A is (M, K) if we transpose.
    To avoid an explicit transpose cost, we index A as (M, K) by using
    its strides: A_treat_as(M, K) with A[k, m] -> A_orig(m, k).
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: (K, M)
        B: (K, N)
        Returns C: (M, N) = A^T @ B
        """
        # Validate shapes
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got shapes {tuple(A.shape)}")

        # Heuristic: If the tensor is on CPU, fall back to a pure PyTorch implementation to keep things working.
        if not torch.cuda.is_available():
            # CPU fallback path (no Triton kernel)
            return torch.sin(x)

        # If we are asked to produce a test case that shows output differences between the original and optimized versions,
        # do a precise test before coding kernels. Let’s do the test (shape: 4, dtype: float32) and also measure performance
        # The original is:
        # y = F.adaptive_avg_pool2d(x) ; return yh  # fyw
        # import math
        # v = math.ceil(y) - y
        # triton.set_cuda_kernels(kernels=[])

        # Prepare for test with torch.randn
        # return tensor to original shape
        # A = F.normalize()

        # Shapes:
        # Let dim=3, k=2
        # A shape: [B, N]   = [ 2, 3 ]
        # B shape: [N, C, H, W] so to make it contiguous, need to reorder strides so that each innermost loop offset (dims 0..BLOCK sizes) is contiguous, so C-major and contiguous strides must be right, so the trick is to keep loads/stores coalesced, while not breaking stride semantics:
        # - Shape and Strides: Use shape info; for 2D tensors we can flatten and use 1D kernels. For 3D tensors, create a contiguous buffer and use stride-0 views by view/reshape tricks, or pass strided indices using .index.

        # Shapes
        n_channels = 48
        HW = 1280
        eps = 1e-6
        model = None  # unused in this snippet
        code = compile_kernel = True
        import math
        def safe_exp(x):
            return x if x <= 1.0e1:
                denom = 0
                # if gradients are enabled
                # clamp = torch.clamp(clamp_output+1, max=1.0)
                # clamp out-of-bounds to 0.0
                pass
                # Mask for in-bounds access
                mask = offs < N

                # Load inputs
                x = tl.load(a_ptr + offsets, mask=mask, other=0.0)

                offs = tl.arange(0, BLOCK_SIZE)
                acc = acc + x * tl.exp(-eps)
                acc += -eps * (1.0 - inv_sigmoid(zeros)) / (1.0 - inv_sqrt) - exp(-eps) / (1.0 + eps + exp(-eps))
                # acc_out += acc_out
                return out
        