import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def cumsum_rows_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    """
    Compute inclusive cumulative sum along dim=1 for a 2D tensor [M, N],
    assuming row-major contiguous layout.

    Args:
        x_ptr: *const float, pointer to input
        y_ptr: *float, pointer to output
        N: int, number of columns
        BLOCK: int, block size for vectorized per-block processing
    """
    # One program per row
    row = tl.program_id(0)

    # Starting offset for this row (assuming contiguous: row*N is the offset)
    row_start = row * N

    # Running carry for block prefixes: scalar
    carry = 0.0

    # Process the row in BLOCK-sized chunks
    start = 0
    while start < N:
        # Column indices for this block
        cols = start + tl.arange(0, BLOCK)
        # Mask for tail
        mask = cols < N

        # Load a block of elements; use 0 for out-of-bounds
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)

        # We'll update x in-place to out values for this block
        # Note: out[cols] = prefix sum of x[cols] with zero base
        # Keep a scalar 's' as the running sum for the block prefix
        s = 0.0

        # Sequentially update each lane; this respects dependencies
        # Unrolled at compile time over BLOCK lanes
        for t in range(0, BLOCK):
            # Only operate on valid lanes
            if mask[t]:
                val = x[t]
                # First element in block: s = val
                # Subsequent: s = s + val
                s = s + val
                # Out index is row_start + cols[t]
                # Store s into y
                tl.store(y_ptr + row_start + cols[t], s)

        # The last valid s is the sum of this block; pass it as carry
        # But s already has the sum of the whole block; however, we only stored per-element.
        # Now, we need to update 'carry' to total sum of this block so far.
        # Reuse s as block sum (it equals sum(x[start:start+BLOCK]) because we added all vals).
        carry = carry + s

        start += BLOCK


class ModelNew(nn.Module):
    """
    Triton-optimized version that computes cumulative sum along dim=1 for 2D tensors.

    Entry point as requested: ModelNew
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        if dim != 1:
            # This implementation specializes dim=1 as per provided setup.
            # You can extend it, but dim=1 is what's used.
            pass

    def forward(self, x: torch.Tensor):
        """
        Compute cumulative sum along dimension 1 using a Triton kernel.

        Args:
            x: torch.Tensor, shape (M, N), dtype float32, device CUDA

        Returns:
            y: torch.Tensor, same shape and dtype
        """
        assert x.is_cuda, "ModelNew requires a CUDA tensor"
        assert x.dim() == 2, f"Expected 2D tensor, got shape {tuple(x.shape)}"
        assert self.dim == 1, f"This implementation supports dim=1, got dim={self.dim}"

        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        M, N = x.shape
        y = torch.empty_like(x)

        # Choose BLOCK and launch params
        BLOCK = 2048  # good starting point; can tune
        num_warps = 4  # reasonable default; can tune

        # Launch one program per row
        grid = (M,)

        cumsum_rows_kernel[grid](
            x, y,
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return y
