import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _cumsum_rows_blocks_kernel(
    x_ptr, out_ptr,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    N,  # number of cols (length to scan)
    BLOCK: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)

    # Create a vector of indices [0..BLOCK)
    offs = tl.arange(0, BLOCK)

    # Pass 1: process blocks, compute block sums, write out initial sums, store block sums
    num_blocks = (N + BLOCK - 1) // BLOCK

    # We'll keep block_sums in registers as a Python list for small num_blocks.
    # Alternative: write to out first pass as a temporary and read back, but here we use registers.
    block_sums = [0.0] * num_blocks

    col_start = 0
    for b in range(0, num_blocks):
        # Compute block start column
        col = col_start  # not needed below but clear
        idx = row * stride_x_row + col_start + offs * stride_x_col
        mask = (col_start + offs) < N

        # Load block; use 0 for out-of-bounds
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)

        # Compute inclusive scan in registers and store to out
        # Also accumulate block sum
        block_sum = 0.0
        out_base = row * stride_out_row + col_start + offs * stride_out_col
        for i in range(0, BLOCK):
            vi = x[i]
            block_sum += vi
            # only store if in bounds
            if i < N - col_start:
                tl.store(out_ptr + out_base + i * stride_out_col, block_sum)
        block_sums[b] = block_sum

        col_start += BLOCK

    # Pass 2: compute exclusive prefix of block sums, then add to out blocks
    # exclusive[b] = sum(block_sums[0..b-1])
    excl = 0.0
    for b in range(0, num_blocks):
        s = block_sums[b]
        block_sums[b] = excl  # now block_sums[b] holds the prefix to add
        excl += s

    # Now add the prefix to each block in out
    col_start = 0
    for b in range(0, num_blocks):
        idx = row * stride_out_row + col_start + offs * stride_out_col
        mask = (col_start + offs) < N
        v = tl.load(out_ptr + idx, mask=mask, other=0.0)
        add_val = block_sums[b]
        v = v + add_val
        tl.store(out_ptr + idx, v, mask=mask)
        col_start += BLOCK


class ModelNew(nn.Module):
    """
    Triton-optimized cumulative sum along a specified dimension.
    Current implementation supports 2D tensors and dim=1 (as in your setup).
    Falls back to torch.cumsum on CPU.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        # Tunable parameters
        self.block_size = 1024
        self.num_warps = 4

    def forward(self, x: torch.Tensor):
        # Fallback for non-CUDA or non-2D or wrong dim
        if not x.is_cuda:
            return torch.cumsum(x, dim=self.dim)

        if x.dim() != 2:
            # Fallback for simplicity; could be extended
            return torch.cumsum(x, dim=self.dim)

        if self.dim != 1:
            # This kernel is written for dim=1; use fallback otherwise
            return torch.cumsum(x, dim=self.dim)

        M, N = x.shape  # rows, cols
        # Allocate output
        out = torch.empty_like(x)

        # Extract strides in elements
        stride_x_row, stride_x_col = x.stride(0), x.stride(1)
        stride_out_row, stride_out_col = out.stride(0), out.stride(1)

        # Launch grid: one program per row
        grid = (M,)

        _cumsum_rows_blocks_kernel[grid](
            x, out,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            N,
            BLOCK=self.block_size,
            num_warps=self.num_warps,
        )

        return out
