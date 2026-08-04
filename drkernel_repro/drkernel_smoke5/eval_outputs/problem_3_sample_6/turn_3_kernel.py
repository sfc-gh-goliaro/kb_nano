import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _scale_kernel(X, OUT, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise scale: OUT[i] = X[i] * alpha for i in [0, n_elements).
    1D launch; masked tail.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0)
    y = x * alpha
    tl.store(OUT + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized version:
      Original: y = (x @ W^T + b) * s + (x @ W^T + b)  => two elementwise passes
      New:      z = x @ W^T + b  (torch/cuBLAS)
                y = z * (1 + s)  (single-pass Triton kernel)
    Falls back to torch if tensor is not CUDA.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

        # Triton kernel launch parameters (tuned for B200 memory bandwidth)
        self.block_size = 4096
        self.num_warps = 8
        self.num_stages = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GEMM + bias via cuBLAS (torch)
        z = self.linear(x)

        # CPU fallback: just do the fused scale in torch
        if not z.is_cuda:
            alpha = 1.0 + self.scaling_factor
            return z * alpha

        # Ensure contiguous for pointer arithmetic
        if not z.is_contiguous():
            z = z.contiguous()

        M = z.numel()
        y = torch.empty_like(z)

        grid = (triton.cdiv(M, self.block_size),)

        _scale_kernel[grid](
            z, y,
            M,
            1.0 + self.scaling_factor,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        return y
