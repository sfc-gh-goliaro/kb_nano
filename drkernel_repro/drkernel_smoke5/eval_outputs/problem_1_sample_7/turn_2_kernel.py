import math
import torch

import triton
import triton.language as tl


@triton.jit
def prefix_sum_kernel(x_ptr, y_ptr, out_ptr,
                      n_elements: tl.constexpr,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)

    # Load x and y; out = x + y
    # Simple elementwise kernel example
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters

    def forward(self, a, b):
        # a: (B, H), b: (B, H), both float
        # Fallback to torch if not CUDA
        if not (a.is_cuda and b.is_cuda):
            return torch.add(a, b)

        return out
