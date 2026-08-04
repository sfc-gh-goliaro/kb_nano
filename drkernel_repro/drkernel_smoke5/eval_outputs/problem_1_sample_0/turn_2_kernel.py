import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: logsumexp along dim=1 (columns) for each row of a 2D tensor
# x: shape (M, N), row-major
# y: shape (M,), output logsumexp per row
@triton.jit
def _logsumexp_rows_kernel(
    x_ptr,               # *const T
    y_ptr,               # *float32
    M: tl.constexpr,     # number of rows
    N: tl.constexpr,     # number of cols
    stride_xm,           # stride for row in elements
    stride_xn,           # stride for col in elements
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)  # row id
    # guard: if pid >= M: return
    # (usually grid = (M,) so not needed, but keep safety)
    if pid >= M:
        return

    row_x_ptr = x_ptr + pid * stride_xm

    # Pass 1: compute row-wise max in float32 for numerical stability
    row_max = -float('inf')
    col = 0
    while col < N:
        offs = col + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_x_ptr + offs * stride_xn, mask=mask, other=-float('inf'))
        x_f32 = x.to(tl.float32)
        local_max = tl.max(tl.where(mask, x_f32, -float('inf')), axis=0)
        row_max = tl.maximum(row_max, local_max)
        col += BLOCK_N

    # Pass 2: compute sum exp(x - row_max) in float32
    row_sum = 0.0
    col = 0
    while col < N:
        offs = col + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_x_ptr + offs * stride_xn, mask=mask, other=-float('inf'))
        x_f32 = x.to(tl.float32)
        z = tl.exp(x_f32 - row_max)
        z = tl.where(mask, z, 0.0)
        local_sum = tl.sum(z, axis=0)
        row_sum += local_sum
        col += BLOCK_N

    # logsumexp = row_max + log(row_sum)
    out = row_max + tl.log(row_sum)
    tl.store(y_ptr + pid, out)


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for:
      y = torch.cumsum(x, dim=0) + bias.view(1, C, 1, 1)

    Notes:
    - Implements logsumexp(x, dim=1) for 2D tensors using a Triton kernel on CUDA.
    - Falls back to torch.logsumexp if Triton/CUDA is not available.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect 2D tensor and dim=1
        if x.dim() != 2 or self.dim != 1:
            # Fallback to PyTorch for non-2D or different dim
            return torch.logsumexp(x, dim=self.dim)

        if not x.is_cuda or not TRITON_AVAILABLE:
            return torch.logsumexp(x, dim=1)

        M, N = x.shape
        # Allocate output (float32 for numeric stability; cast back at end if needed)
        y = torch.empty((M,), device=x.device, dtype=torch.float32)

        # Compute strides in elements
        stride_xm = x.stride(0)
        stride_xn = x.stride(1)

        # Choose block size: power-of-two up to 1024
        BLOCK_N = 1 << int(math.ceil(math.log2(max(1, N)))) if N > 0 else 1
        BLOCK_N = min(BLOCK_N, 1024)

        # Grid: one program per row
        grid = (M,)

        # Launch kernel
        _logsumexp_rows_kernel[grid](
            x, y,
            M, N,
            stride_xm, stride_xn,
            BLOCK_N=BLOCK_N,
            num_warps=4,  # heuristic
        )

        # Match PyTorch dtype: return same dtype as input
        if x.dtype != torch.float32:
            return y.to(x.dtype)
        return y
