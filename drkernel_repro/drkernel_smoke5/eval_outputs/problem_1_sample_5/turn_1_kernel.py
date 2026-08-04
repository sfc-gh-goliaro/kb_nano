import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def cumsum_rows_kernel(
    X, Y,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)

    # Running accumulator for the scan
    acc = 0.0

    # Loop over the row in BLOCK_N chunks
    i = 0
    while i < N:
        offs = i + tl.arange(0, BLOCK_N)
        mask = offs < N

        x_ptrs = X + row * stride_xm + offs * stride_xn
        vals = tl.load(x_ptrs, mask=mask, other=0.0)

        # Sequentially accumulate and store per-lane
        # No inter-lane dependencies here.
        j = 0
        while j < BLOCK_N:
            idx = i + j
            if idx < N:
                v = vals[j]
                acc = acc + v
                y_ptr = Y + row * stride_ym + idx * stride_yn
                tl.store(y_ptr, acc)
            j += 1

        i += BLOCK_N


class ModelNew(nn.Module):
    """
    Triton-optimized version that computes cumulative sum along a specified dimension.
    Currently supports 2D tensors and dim in {0, 1}. For other dims, falls back to torch.cumsum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # Fallback for non-CUDA
        if not x.is_cuda:
            return torch.cumsum(x, dim=self.dim)

        # Only handle 2D + dim in {0,1} in the Triton kernel; else fallback
        if x.dim() != 2 or self.dim not in (0, 1):
            return torch.cumsum(x, dim=self.dim)

        # Ensure contiguous for best performance (still works with non-contiguous but slower)
        if not x.is_contiguous():
            x = x.contiguous()

        # We will compute along the last dimension by permuting if needed
        if self.dim == 0:
            # Swap rows/cols view without copy: make it shape (N, M) logically
            # Easiest is to use as_strided, but for simplicity, permute and make contiguous
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
        )
        return y
