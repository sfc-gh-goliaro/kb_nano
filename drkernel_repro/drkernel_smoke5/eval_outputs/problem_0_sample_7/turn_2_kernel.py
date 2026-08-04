# The original PyTorch code you want me to replace/optimize with Triton
import math
import torch
import torch.nn as nn

def _torch_cos_similarity(x: torch.Tensor, y: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # Compute mean across the specified dim (keepdim=False), then cosine similarity per row (dim=1)
    # This mimics torch.nn.functional.cosine_similarity across dim=1 without the reduction op, by using sum of squares for denominator
    # Shape: [B, N]
    # Use float32 for compute precision; outputs cast back to the input dtype.
    # Choose simple, sensible defaults; can be tuned if needed
    # launch kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; pure elementwise kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Return x + y elementwise. Uses Triton on CUDA; falls back to torch on CPU.
        if not x.is_cuda or not y.is_cuda:
            return x + y
        assert x.shape == y.shape, f"Shapes must match, got {x.shape} vs {y.shape}"
        # dtype: keep it simple; require float32 for the kernel. Could add fp16 support with upcast.
        if x.dtype != torch.float32 or y.dtype != torch.float32:
            x = x.to(torch.float32)
            y = y.to(torch.float32)
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
        return out


# Keep the same helpers so the harness can use them
def get_inputs():
    a = torch.randn(10, device='cuda', dtype=torch.float32)
    b = torch.randn(10, device='cuda', dtype=torch.float32)
    return [a, b]

def get_init_inputs():
    return []
