import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _fused_add_softmax_tanh_scale_2d_kernel(
    x_ptr,             # *float32, shape [B, C, H, W] contiguous
    bias_ptr,          # *float32, shape [C]
    y_ptr,             # *float32, shape [B, C, H, W] contiguous
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    scaling: tl.float32,
    BLOCK_C: tl.constexpr,
):
    # program ids
    pid_bhw = tl.program_id(0)
    pid_cb  = tl.program_id(1)

    # decode (b, h, w) from pid_bhw
    w = pid_bhw % W
    tmp = pid_bhw // W
    h = tmp % H
    b = tmp // H

    # channel block start
    cb_start = pid_cb * BLOCK_C
    offs_c = cb_start + tl.arange(0, BLOCK_C)
    mask = offs_c < C

    # base offset for (b, h, w): since layout is [B, C, H, W] contiguous
    # index formula: i = ((b * C + c) * H + h) * W + w
    base = ((b * C) * H + h) * W + w

    # compute full channel index for each lane
    idx = (offs_c * H + h) * W + w
    ptrs = x_ptr + base + offs_c * (H * W)

    # load x, add bias
    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    # bias is 1D over C
    bias = tl.load(bias_ptr + offs_c, mask=mask, other=0.0)
    x = x + bias

    # stable softmax over this block: need global max over C
    # first pass: block max
    is_valid = mask & (x > -float('inf'))
    block_max = tl.max(tl.where(is_valid, x, -float('inf')), axis=0)
    # second pass: sum of exp(x - block_max)
    x_shift = x - block_max
    exp_x = tl.exp(x_shift)
    sum_exp = tl.sum(tl.where(is_valid, exp_x, 0.0), axis=0)
    # third pass: normalize
    prob = exp_x / sum_exp  # only valid lanes have finite values

    # apply tanh and scaling
    # tanh in float32 is fine
    tanh_prob = tl.tanh(prob)
    out = tanh_prob * scaling

    # store
    out_ptrs = y_ptr + base + offs_c * (H * W)
    tl.store(out_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version that keeps ConvTranspose3d + mean in PyTorch,
    and fuses (add bias) + softmax (over channels) + tanh + scale into a single Triton kernel.
    Entry point class name: ModelNew
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Keep bias as in original: shape [1, C, 1, 1, 1], will squeeze to [C] for kernel
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # 1) ConvTranspose3d
        x = self.conv_transpose(x)  # [B, C, D, H, W]
        # 2) Mean over depth -> [B, C, 1, H, W]
        x = x.mean(dim=2, keepdim=True)

        # If not CUDA, fall back to PyTorch for rest
        if not x.is_cuda:
            # Original sequence
            x = x + self.bias
            x = torch.softmax(x, dim=1)
            x = torch.tanh(x)
            x = x * self.scaling_factor
            return x

        # Ensure contiguous and view as [B, C, H, W] (D=1 collapsed)
        x = x.contiguous()
        B, C, D, H, W = x.shape
        assert D == 1, f"Expected D=1 after mean, got D={D}"
        x_2d = x.view(B, C, H, W).contiguous()

        # Prepare output
        y = torch.empty_like(x_2d)

        # Bias: squeeze to [C] and ensure dtype/device
        bias = self.bias
        # bias shape [1, C, 1, 1, 1] -> [C]
        bias_1d = bias.view(-1).contiguous()
        # Cast to float32 for numeric stability in kernel
        x_2d_f32 = x_2d.float()
        bias_f32 = bias_1d.float()
        y_f32 = y.float()

        # Grid: (B*H*W, ceil_div(C, BLOCK_C))
        BLOCK_C = 128
        grid = (B * H * W, triton.cdiv(C, BLOCK_C))

        _fused_add_softmax_tanh_scale_2d_kernel[grid](
            x_2d_f32, bias_f32, y_f32,
            B, C, H, W,
            self.scaling_factor,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )

        # Cast back to original dtype if needed
        if y.dtype != x.dtype:
            y = y.to(x.dtype)

        # View back to [B, C, 1, H, W]
        y = y.view(B, C, 1, H, W)
        return y
