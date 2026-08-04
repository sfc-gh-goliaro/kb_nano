import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _scale_inplace_kernel(x_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    x = x * alpha
    tl.store(x_ptr + offs, x, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model.

    Changes:
    - Removes the redundant clone.
    - Fuses mul+add into a single scale: out = (1 + scaling_factor) * (x @ W^T + b).
    - Uses a tiny Triton kernel to perform the scale in one pass (inference fast path).
    - Falls back to pure PyTorch for autograd (training) to keep gradients correct.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        # Keep Linear to leverage cuBLAS for GEMM
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor):
        # GEMM: y = x @ W^T + b
        y = self.linear(x)

        # alpha = 1 + scaling_factor
        alpha = 1.0 + self.scaling_factor

        # If we're on CUDA and not requiring gradients, use the fast Triton scale kernel
        if y.is_cuda and not y.requires_grad:
            # Ensure contiguous for simple 1D addressing
            if not y.is_contiguous():
                y = y.contiguous()
            n_elements = y.numel()

            # Choose a reasonable block size
            BLOCK = 1024
            grid = (triton.cdiv(n_elements, BLOCK),)

            # Launch kernel: in-place scale
            _scale_inplace_kernel[grid](
                y,  # we modify y in-place
                n_elements,
                alpha,
                BLOCK=BLOCK,
            )
            return y
        else:
            # Fallback: pure PyTorch, keeps autograd
            return y * alpha


# The rest (helpers) can remain the same as in your snippet
batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
