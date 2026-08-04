import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Minimal 1D elementwise square kernel: y[i] = x[i] * x[i]
@triton.jit
def _square_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = x * x
    tl.store(Y + offs, y, mask=mask)


def _triton_square(x: torch.Tensor) -> torch.Tensor:
    """
    Compute elementwise square using Triton.

    Args:
        x: 1D or contiguous tensor on CUDA.

    Returns:
        y: tensor with same shape and dtype as x.
    """
    assert x.is_cuda, "Input must be CUDA for Triton kernel."
    # Ensure 1D contiguous view
    if x.dim() != 1:
        x = x.view(-1)
    if not x.is_contiguous():
        x = x.contiguous()

    y = torch.empty_like(x)
    n = x.numel()

    # Configure launch
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    _square_kernel[grid](
        x, y,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    # Restore original shape if needed
    return y.view_as(x)


class ModelNew(nn.Module):
    """
    Triton-optimized version that computes y = x * x elementwise.

    Falls back to torch.square if Triton/CUDA is not available.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # Fallback to torch if not CUDA or Triton missing
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            return torch.square(x)
        # dtype guard: support common floating types
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return torch.square(x)
        return _triton_square(x)


# If you need to mirror the helper functions:
def get_inputs():
    # Example: large 1D tensor on CUDA for benchmarking
    return [torch.rand(1_000_000, device='cuda')]

def get_init_inputs():
    return []
