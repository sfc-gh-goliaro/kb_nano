import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _cumsum_2d_axis1_kernel(
    x_ptr,               # *T
    y_ptr,               # *T
    M,                   # rows
    N,                   # cols
    stride_x0,           # row stride (elements)
    stride_x1,           # col stride (elements)
    stride_y0,
    stride_y1,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    if row >= M:
        return

    # Running carry (scalar) in the tensor's dtype
    # We'll initialize after first element to keep dtype alignment.
    # Start from col = 0
    col = 0
    # Handle empty row
    if N == 0:
        return

    # Load first element to set accumulator dtype
    first = tl.load(x_ptr + row * stride_x0 + 0 * stride_x1)
    acc = first
    # Store y[0] = x[0]
    tl.store(y_ptr + row * stride_y0 + 0 * stride_y1, acc)

    col = 1

    # Process the rest
    while col < N:
        # Optional block chunk: not necessary but can reduce loop iterations
        # We will do sequential per-element for simplicity and correctness.
        # If you want to vectorize, you must do a vector prefix-scan inside the block,
        # but that changes FMA order. Here we keep strict order.
        i = 0
        while i < BLOCK_SIZE and col < N:
            # load scalar
            val = tl.load(x_ptr + row * stride_x0 + col * stride_x1)
            acc = acc + val
            # store scalar
            tl.store(y_ptr + row * stride_y0 + col * stride_y1, acc)
            col += 1
            i += 1
        # After a full BLOCK_SIZE-step, 'col' already advanced; loop continues if needed.


def _triton_cumsum_2d_axis1(x: torch.Tensor) -> torch.Tensor:
    """
    Cumulative sum along dim=1 for a 2D CUDA tensor using a simple, order-preserving Triton kernel.
    """
    assert x.is_cuda, "Input must be CUDA for Triton kernel."
    assert x.dim() == 2, f"Expected 2D tensor, got shape {tuple(x.shape)}"

    # Contiguous for coalesced access
    if not x.is_contiguous():
        x = x.contiguous()

    M, N = x.shape
    y = torch.empty_like(x)

    stride_x0, stride_x1 = x.stride(0), x.stride(1)
    stride_y0, stride_y1 = y.stride(0), y.stride(1)

    BLOCK_SIZE = 256
    grid = (M,)

    _cumsum_2d_axis1_kernel[grid](
        x, y,
        M, N,
        stride_x0, stride_x1,
        stride_y0, stride_y1,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized cumulative sum model.
    Computes cumsum along a specified dimension. Fast path for 2D tensors on CUDA.
    Falls back to torch.cumsum otherwise.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        d = self.dim
        if d < 0:
            d = d + x.dim()
        if not (0 <= d < x.dim()):
            raise IndexError(f"dim out of range (got {self.dim} for tensor of dim {x.dim()})")

        # Fast path: 2D + CUDA + Triton available
        if x.dim() == 2 and x.is_cuda and TRITON_AVAILABLE:
            if d == 1:
                return _triton_cumsum_2d_axis1(x)
            elif d == 0:
                # Cumsum over rows: transpose, compute, transpose back
                xt = x.t().contiguous()   # (N, M)
                yt = _triton_cumsum_2d_axis1(xt)  # cumsum over columns of xt
                y = yt.t().contiguous()          # back to (M, N)
                return y
            else:
                # Should not happen for 2D, but just in case
                return torch.cumsum(x, dim=d)

        # Fallback for CPU, non-2D, or no Triton
        return torch.cumsum(x, dim=d)


# Helpers to mirror the original interface
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    # Put tensor on CUDA to exercise Triton path
    return [torch.rand(batch_size, *input_shape, device='cuda')]

def get_init_inputs():
    return [dim]
