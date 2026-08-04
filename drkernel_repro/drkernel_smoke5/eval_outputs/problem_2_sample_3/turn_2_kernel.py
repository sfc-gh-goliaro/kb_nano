import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def tanh_scale_bias_kernel(
    x_ptr,          # *const T
    y_ptr,          # *T
    n_elements: tl.constexpr,
    scale,          # float32
    bias,           # float32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load as float32 for stable math
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Fused multiply-add
    z = x * scale + bias

    # Numerically stable tanh via sigmoid form using exp(-2*|z|)
    az = tl.abs(z)
    t = tl.exp(-2.0 * az)              # in (0, 1]
    num = 1.0 - t
    den = 1.0 + t
    tanh_abs = num / den                # in [0,1]
    sign = tl.where(z >= 0, 1.0, -1.0)
    y = sign * tanh_abs

    # Store; Triton will cast to destination dtype if needed
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Drop-in replacement for the original Model:
    forward(x) computes tanh(x * scale + bias) elementwise.
    Uses Triton on CUDA for speed; falls back to torch ops otherwise.
    """
    def __init__(self, scale: float, bias: float):
        super().__init__()
        # Keep as Python floats; kernel will receive as float32 scalars
        self.scale = float(scale)
        self.bias = float(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to torch if not CUDA or Triton not available
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            return torch.tanh(x * self.scale + self.bias)

        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        # Allocate output
        y = torch.empty_like(x)

        # Flatten to 1D for simple elementwise kernel
        n = x.numel()

        # Choose block size and launch config
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)

        # Launch kernel
        tanh_scale_bias_kernel[grid](
            x, y,
            n,
            self.scale, self.bias,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
            num_stages=2,
        )

        return y
