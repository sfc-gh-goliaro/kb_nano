import math
import torch
import torch.nn as nn

# Try importing Triton; fall back gracefully if unavailable.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def cumsum_rows_kernel(
    X, Y,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    """
    Compute inclusive cumulative sum along the last dimension for a 2D tensor (M rows, N cols).
    One program per row. Iterate over the row in BLOCK_N-sized chunks.
    """
    row = tl.program_id(0)

    # Running accumulator (float32)
    acc = 0.0

    i = 0
    while i < N:
        # Process up to BLOCK_N elements in this chunk, one by one to avoid inter-lane dependencies.
        # This loop is unrolled at compile time since BLOCK_N is constexpr.
        for j in range(0, BLOCK_N):
            idx = i + j
            if idx < N:
                x_ptr = X + row * stride_xm + idx * stride_xn
                v = tl.load(x_ptr).to(tl.float32)
                acc = acc + v
                y_ptr = Y + row * stride_ym + idx * stride_yn
                tl.store(y_ptr, acc)
        i += BLOCK_N


class ModelNew(nn.Module):
    """
    Triton-optimized version that computes cumulative sum along a specified dimension.
    Supports 2D tensors and dim in {0, 1} on CUDA. Falls back to torch.cumsum otherwise.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # Fallback for non-CUDA or missing Triton
        if (not _HAS_TRITON) or (not x.is_cuda):
            return torch.cumsum(x, dim=self.dim)

        # Only handle 2D + dim in {0,1} in the Triton kernel; else fallback
        if x.dim() != 2 or self.dim not in (0, 1):
            return torch.cumsum(x, dim=self.dim)

        # Ensure contiguous for best performance
        if not x.is_contiguous():
            x = x.contiguous()

        # We will compute along the last dimension by permuting if needed
        if self.dim == 0:
            # Swap to treat logical shape (N, M) = (x.size(1), x.size(0))
            x_t = x.transpose(0, 1).contiguous()
            out_t = torch.empty_like(x_t)

            M, N = x_t.shape
            stride_xm, stride_xn = x_t.stride(0), x_t.stride(1)
            stride_ym, stride_yn = out_t.stride(0), out_t.stride(1)

            BLOCK = 1024
            grid = (M,)
            cumsum_rows_kernel[grid](
                x_t, out_t,
                M, N,
                stride_xm, stride_xn,
                stride_ym, stride_yn,
                BLOCK_N=BLOCK,
                num_warps=4,
                num_stages=2,
            )
            out = out_t.transpose(0, 1).contiguous()
            return out

        # dim == 1: natural 2D, shape (M, N)
        M, N = x.shape
        y = torch.empty_like(x)

        # Strides in elements
        stride_xm, stride_xn = x.stride(0), x.stride(1)
        stride_ym, stride_yn = y.stride(0), y.stride(1)

        BLOCK = 1024
        grid = (M,)
        cumsum_rows_kernel[grid](
            x, y,
            M, N,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            BLOCK_N=BLOCK,
            num_warps=4,
            num_stages=2,
        )
        return y
