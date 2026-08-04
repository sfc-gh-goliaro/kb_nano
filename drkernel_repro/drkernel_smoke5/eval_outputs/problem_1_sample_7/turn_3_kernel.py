import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _cumsum_rows_blocks_kernel(
    x_ptr, out_ptr, block_sums_ptr,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    N,  # number of cols to scan
    BLOCK: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)

    # Column offsets for a block
    offs = tl.arange(0, BLOCK)

    # Pass 1: process blocks, compute block sums, write out initial sums, store block sums
    num_blocks = (N + BLOCK - 1) // BLOCK

    # Loop over blocks
    for b in range(0, num_blocks):
        col = b * BLOCK
        # Compute pointers for this block
        x_idx = row * stride_x_row + col + offs * stride_x_col
        out_idx = row * stride_out_row + col + offs * stride_out_col
        mask = (col + offs) < N

        # Load block; use 0 for out-of-bounds
        x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

        # Inclusive scan in registers, store to out, accumulate block sum
        s = 0.0
        for i in range(0, BLOCK):
            vi = x[i]
            s = s + vi
            # Store s to out at position (row, col + i)
            tl.store(out_ptr + out_idx + i * stride_out_col, s, mask=mask[0] and (col + i < N))
        # Store block sum for this block
        tl.store(block_sums_ptr + b, s)

    # Pass 2: compute exclusive prefix of block sums, then add to out blocks
    # excl = sum of previous blocks; block_sums[b] holds sum of block b
    excl = 0.0
    for b in range(0, num_blocks):
        t = tl.load(block_sums_ptr + b)
        # set block_sums[b] = excl (the value to add to this block)
        tl.store(block_sums_ptr + b, excl)
        excl = excl + t

    # Now add the prefix to each block in out
    for b in range(0, num_blocks):
        col = b * BLOCK
        out_idx = row * stride_out_row + col + offs * stride_out_col
        mask = (col + offs) < N
        v = tl.load(out_ptr + out_idx, mask=mask, other=0.0)
        add_val = tl.load(block_sums_ptr + b)
        v = v + add_val
        tl.store(out_ptr + out_idx, v, mask=mask)


class ModelNew(nn.Module):
    """
    Triton implementation of cumulative sum along dim=1 for 2D tensors.
    Falls back to torch.cumsum on CPU or non-float32.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        # Tunable parameters
        self.block_size = 1024
        self.num_warps = 4

    def forward(self, x: torch.Tensor):
        # Fallbacks
        if not x.is_cuda:
            return torch.cumsum(x, dim=self.dim)
        if x.dim() != 2:
            return torch.cumsum(x, dim=self.dim)
        if self.dim != 1:
            return torch.cumsum(x, dim=self.dim)
        if x.dtype != torch.float32:
            return torch.cumsum(x, dim=self.dim)

        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        M, N = x.shape  # rows, cols
        out = torch.empty_like(x)

        # Strides in elements
        stride_x_row, stride_x_col = x.stride(0), x.stride(1)
        stride_out_row, stride_out_col = out.stride(0), out.stride(1)

        # Temporary for block sums (one per block per row)
        num_blocks = (N + self.block_size - 1) // self.block_size
        block_sums = torch.empty((M, num_blocks), device=x.device, dtype=x.dtype)

        # Launch grid: one program per row
        grid = (M,)

        _cumsum_rows_blocks_kernel[grid](
            x, out, block_sums,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            N,
            BLOCK=self.block_size,
            num_warps=self.num_warps,
        )

        return out
