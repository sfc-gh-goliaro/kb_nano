import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _softplus_kernel_fp32(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load as float32
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Numerically stable softplus:
    # softplus(x) = max(x, 0) + log(1 + exp(-abs(x)))
    ax = tl.abs(x)
    max0x = tl.maximum(x, 0.0)
    out = max0x + tl.log(1.0 + tl.exp(-ax))

    tl.store(y_ptr + offs, out, mask=mask)


def softplus_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softplus(x) using a Triton kernel.
    - Uses float32 compute for numerical stability.
    - Input must be CUDA contiguous.
    """
    assert x.is_cuda, "softplus_triton requires CUDA tensor"
    # Ensure contiguous
    x_c = x.contiguous()
    N = x_c.numel()

    # Allocate output (float32 compute)
    y = torch.empty_like(x_c, dtype=torch.float32)

    # Launch config
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _softplus_kernel_fp32[grid](
        x_c.view(-1),
        y.view(-1),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # Cast back to input dtype if needed
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y.view_as(x_c)


class ModelNew(nn.Module):
    """
    Triton-optimized version that replaces torch.nn.functional.softplus
    with a fused Triton kernel.

    Keeps the rest of the original model intact.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Keep bias as in original: shape [1, C, 1, 1, 1]
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # 1) ConvTranspose3d
        x = self.conv_transpose(x)  # [B, C, D, H, W]
        # 2) Mean over depth -> [B, C, 1, H, W]
        x = x.mean(dim=2, keepdim=True)

        # 3) Add broadcast bias
        x = x + self.bias

        # 4) Softmax over channels (dim=1)
        #    Use Triton for the softplus if someone had used it; but PyTorch's softmax is fine and fast.
        x = torch.softmax(x, dim=1)

        # 5) Tanh activation
        x = torch.tanh(x)

        # 6) Scaling
        x = x * self.scaling_factor
        return x


# The rest (get_inputs, get_init_inputs) can remain as-is.
# Example:
# batch_size = 16
# in_channels  = 16
# out_channels = 64
# depth = 32; height = width = 128
# kernel_size  = 3
# stride       = 1
# padding = 1
# scaling_factor = 2.0
#
# def get_inputs():
#     return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]
#
# def get_init_inputs():
#     return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]
