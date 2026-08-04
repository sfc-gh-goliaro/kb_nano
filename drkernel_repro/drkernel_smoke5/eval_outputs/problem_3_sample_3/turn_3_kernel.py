import math
import torch
import torch.nn as nn

# Try to import Triton. If unavailable, we will fall back to PyTorch ops.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _ceil_div(a, b):
    return (a + b - 1) // b


# Simple elementwise kernel: out[i] = 2 * x[i] + 3
if _HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def _affine_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # Compute in same dtype as x to match PyTorch numerics
        y = x * 2.0 + 3.0
        tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model:
    - On CUDA: runs a single-pass elementwise Triton kernel out = 2*x + 3
    - On CPU or if Triton isn't available: falls back to torch ops
    """
    def __init__(self):
        super().__init__()
        # No parameters; behavior matches original forward: y = 2*x + 3
        # If you need to generalize, add alpha, beta.
        self.alpha = 2.0
        self.beta = 3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if:
        # - Not CUDA
        # - Not floating tensor
        if (not _HAS_TRITON) or (not x.is_cuda) or (not x.dtype.is_floating_point):
            # Pure PyTorch path
            return x * self.alpha + self.beta

        # Ensure contiguous
        x_c = x.contiguous()
        out = torch.empty_like(x_c)

        # Flatten to 1-D for simple 1D kernel
        x_flat = x_c.view(-1)
        out_flat = out.view(-1)
        n = x_flat.numel()

        # Launch kernel
        # Grid: 1D over blocks of BLOCK_SIZE elements
        def grid(meta):
            return (_ceil_div(n, meta['BLOCK_SIZE']),)

        _affine_kernel[grid](x_flat, out_flat, n_elements=n)
        return out
