import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _cumsum_rows_kernel(
    x_ptr,          # *const T
    y_ptr,          # *T
    M,              # int: number of rows
    N,              # int: number of cols
    stride_m,       # int: row stride in elements
    stride_n,       # int: col stride in elements
    BLOCK: tl.constexpr,  # tile size (power of two)
    LOG2: tl.constexpr,   # log2(BLOCK)
):
    # Program processes one row
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK)

    # Running carry across tiles for this row (scalar); keep in x dtype
    # We'll infer dtype from loads; start from 0
    carry = 0.0

    start = 0
    while start < N:
        offs = start + cols
        mask = offs < N

        # Compute pointers for this tile
        ptrs = x_ptr + row * stride_m + offs * stride_n

        # Load values; out-of-bounds => 0
        vals = tl.load(ptrs, mask=mask, other=0.0)

        # Add per-row carry to all positions before scan
        vals = vals + carry

        # In-register inclusive scan via Hillis–Steele
        # v = sum_{d=0}^{LOG2-1} (vals shifted right by 2^d)
        d = 0
        while d < LOG2:
            shift = 1 << d  # 2^d

            # s = where(cols >= shift, vals, 0)     -> includes v[l-shift] where l>=shift
            # But that also includes v[0] at positions < shift via padding.
            # We need pure shifted component for this step: spure = s - where(cols >= 2*shift, vals, 0)
            cond_shift = cols >= shift
            cond_double = cols >= (2 * shift)

            s_tmp = tl.where(cond_shift, vals, 0.0)
            s_dble = tl.where(cond_double, vals, 0.0)
            s = s_tmp - s_dble

            vals = vals + s
            d += 1

        # Store results for this tile
        out_ptrs = y_ptr + row * stride_m + offs * stride_n
        tl.store(out_ptrs, vals, mask=mask)

        # Update carry: last valid element of this tile
        # Find the last in-range index: BLOCK-1 or max index where mask True.
        # Simple: use BLOCK-1 and, if it's OOB, set carry = 0.
        last_col = BLOCK - 1
        in_last = (start + last_col) < N
        carry = tl.where(in_last, vals[last_col], 0.0)

        start += BLOCK


class ModelNew(nn.Module):
    """
    Triton-optimized version of torch.cumsum along dim=1 for 2D CUDA tensors.
    Falls back to torch.cumsum otherwise.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Tuning parameters
        self._block = 1024
        self._num_warps = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallbacks and validations
        if not x.is_cuda:
            return torch.cumsum(x, dim=self.dim)
        if x.dim() != 2 or self.dim != 1:
            return torch.cumsum(x, dim=self.dim)

        # Ensure layout
        if not x.is_contiguous():
            x = x.contiguous()

        M, N = x.shape
        y = torch.empty_like(x)

        # Strides in elements
        stride_m = x.stride(0)
        stride_n = x.stride(1)

        # Launch configuration: one program per row
        grid = (M,)

        # Choose BLOCK as power-of-two <= N, default to self._block
        # Cap to 1024 to be safe
        block = min(self._block, 1 << int(math.floor(math.log2(max(N, 1)))))
        log2_block = int(math.log2(block))

        _cumsum_rows_kernel[grid](
            x, y,
            M, N,
            stride_m, stride_n,
            BLOCK=block,
            LOG2=log2_block,
            num_warps=self._num_warps,
        )

        return y
