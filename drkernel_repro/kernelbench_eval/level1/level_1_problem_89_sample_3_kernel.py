import torch
import triton
import triton.language as tl


@triton.jit
def cumsum_rows_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    """
    Inclusive cumulative sum along dim=1 for a 2D tensor [M, N] (row-major contiguous).
    Parallel, block-wise scan per row.

    Args:
        x_ptr: *const float32
        y_ptr: *float32
        N: int (number of columns)
        BLOCK: int (tile width, power of two preferred)
    """
    # One program per row
    row = tl.program_id(0)
    row_start = row * N

    # Running carry of sum of all previous tiles for this row
    carry = 0.0

    start = 0
    while start < N:
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N

        # Load tile with mask; out-of-bounds as 0
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)

        # Vector of lane indices
        idx = tl.arange(0, BLOCK)

        # Compute inclusive scan within the tile using Hillis–Steele
        # working vector 'tmp' starts as x
        tmp = x
        offset = 1
        while offset < BLOCK:
            # shifted = tmp[idx - offset] if idx >= offset else 0
            shifted = tl.where(idx >= offset, tmp[idx - offset], 0.0)
            tmp = tmp + shifted
            offset *= 2

        # The above 'tmp' is inclusive scan of x.
        # Exclusive scan within tile: s = tmp - x  (except lane 0 which becomes 0)
        s = tmp - x

        # Store exclusive scan result for this tile (pre-carry); add carry on Python side after next step
        # But we need to store post-carry values: s + carry
        out = s + carry
        tl.store(y_ptr + row_start + cols, out, mask=mask)

        # Update carry: carry += sum of this tile's x
        # Reduce sum over vector lanes
        sum_tile = 0.0
        for t in range(0, BLOCK):
            sum_tile += x[t]
        carry = carry + sum_tile

        start += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallbacks and checks
        if not x.is_cuda:
            raise AssertionError("ModelNew requires CUDA tensor")
        if x.dtype != torch.float32:
            raise AssertionError("ModelNew currently supports float32 only")
        if x.dim() != 2:
            raise AssertionError(f"Expected 2D tensor, got shape {tuple(x.shape)}")
        if self.dim != 1:
            raise AssertionError(f"This implementation supports dim=1, got dim={self.dim}")

        if not x.is_contiguous():
            x = x.contiguous()

        M, N = x.shape
        y = torch.empty_like(x)

        BLOCK = 1024  # power of two; good starting point
        grid = (M,)

        cumsum_rows_kernel[grid](
            x, y,
            N,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return y
