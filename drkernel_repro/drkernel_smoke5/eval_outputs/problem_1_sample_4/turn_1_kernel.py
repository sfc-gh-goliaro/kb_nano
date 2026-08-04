import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: cumulative sum along dim=1 for a 2D tensor (M, N), one program per row.
@triton.jit
def _cumsum_2d_axis1_kernel(
    x_ptr,               # *const T
    y_ptr,               # *T
    M: tl.constexpr,     # number of rows
    N: tl.constexpr,     # number of cols
    stride_x0,           # row stride for x (in elements)
    stride_x1,           # col stride for x (in elements)
    stride_y0,           # row stride for y (in elements)
    stride_y1,           # col stride for y (in elements)
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    # guard: if grid is larger than M (not here), early return
    if row >= M:
        return

    # column offsets for a block
    offs = tl.arange(0, BLOCK_SIZE)

    # running accumulator in the same dtype as x
    # We'll create it as a scalar; Triton will handle casting on ops with vector values.
    # Initialize from the first element to keep dtype alignment with x.
    first = tl.load(x_ptr + row * stride_x0 + 0 * stride_x1)
    acc = first

    col = 0
    while col < N:
        idx = col + offs
        mask = idx < N
        # load a block of values; "other=0" for out-of-bounds
        vals = tl.load(x_ptr + row * stride_x0 + idx * stride_x1, mask=mask, other=0)
        # inclusive scan within the block, sequentially:
        # Note: we must do scalar ops to preserve dtype and numerical parity.
        for i in range(BLOCK_SIZE):
            vi = vals[i]
            # if masked, vi was 0; still fine.
            acc = acc + vi
            # store result to y
            tl.store(y_ptr + row * stride_y0 + (col + i) * stride_y1, acc)
        col += BLOCK_SIZE
        # next block: acc is carry from previous block


def _triton_cumsum_2d_axis1(x: torch.Tensor) -> torch.Tensor:
    """
    Compute cumulative sum along dim=1 for a 2D CUDA tensor using Triton.

    Args:
        x: (M, N) tensor on CUDA device.

    Returns:
        y: (M, N) cumulative sum along axis 1.
    """
    assert x.is_cuda, "Input must be a CUDA tensor for Triton implementation."
    assert x.dim() == 2, f"Expected 2D tensor, got shape {tuple(x.shape)}"

    # Make contiguous for best performance (still pass strides in case)
    if not x.is_contiguous():
        x = x.contiguous()

    M, N = x.shape
    y = torch.empty_like(x)

    # Extract element-wise strides (in elements, not bytes)
    stride_x0, stride_x1 = x.stride(0), x.stride(1)
    stride_y0, stride_y1 = y.stride(0), y.stride(1)

    # Choose meta-parameters
    BLOCK_SIZE = 256
    num_warps = 4

    grid = (M,)

    _cumsum_2d_axis1_kernel[grid](
        x, y,
        M, N,
        stride_x0, stride_x1,
        stride_y0, stride_y1,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of the cumulative sum model.
    Computes torch-like cumulative sum along a specified dimension.
    Currently supports 2D tensors and dim in {0, 1}. Falls back to torch.cumsum on CPU.

    Entry point name: ModelNew
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # Normalize dim
        d = self.dim
        if d < 0:
            d = d + x.dim()
        if not (0 <= d < x.dim()):
            raise IndexError(f"dim out of range (got {self.dim} for tensor of dim {x.dim()})")

        # Only 2D is implemented in Triton fast path; fall back otherwise
        if x.dim() != 2:
            return torch.cumsum(x, dim=self.dim)

        # If not CUDA or Triton not available, fallback
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            return torch.cumsum(x, dim=d)

        # We implement fast path for dim == 1 (columns). For dim == 0, transpose,
        # compute, then transpose back. This keeps the kernel simple and fast.
        if d == 1:
            return _triton_cumsum_2d_axis1(x)
        elif d == 0:
            # Compute cumsum over rows by operating on the transposed view
            # Note: transpose returns a view; make it contiguous for kernel.
            xt = x.t().contiguous()  # shape (N, M)
            yt = _triton_cumsum_2d_axis1(xt)  # cumsum over columns of xt = over rows of x
            y = yt.t().contiguous()          # back to (M, N)
            return y
        else:
            # Should not happen for 2D, but just in case
            return torch.cumsum(x, dim=d)


# The following helpers mirror the original for convenience/testing.

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    dim = 1
    return [torch.rand(batch_size, *input_shape, device='cuda')]

def get_init_inputs():
    dim = 1
    return [dim]
