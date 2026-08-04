import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _fused_add_softmax_tanh_scale_2d_kernel(
    x_ptr,             # *float, shape [B, C, H, W] contiguous (we'll cast to f32 inside)
    bias_ptr,          # *float, shape [C]
    y_ptr,             # *float, shape [B, C, H, W] contiguous
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    scaling: tl.float32,
    BLOCK_C: tl.constexpr,
):
    # Program ids
    pid_bhw = tl.program_id(0)
    pid_cb  = tl.program_id(1)

    # Decode (b, h, w) from pid_bhw
    w = pid_bhw % W
    t = pid_bhw // W
    h = t % H
    b = t // H

    # Channel block
    cb_start = pid_cb * BLOCK_C
    offs_c = cb_start + tl.arange(0, BLOCK_C)
    mask = offs_c < C

    # Base offset for (b, h, w) in layout [B, C, H, W]
    # index = (((b * C + c) * H + h) * W + w)
    base = ((b * C) * H + h) * W + w
    ptrs = x_ptr + base + offs_c * (H * W)

    # Load x and bias; upcast to float32 for stability
    x = tl.load(ptrs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    bias = tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)

    # Add bias (before softmax, to match original order)
    x = x + bias

    # Invalidate invalid lanes so they don't affect max/sum
    neg_inf = -float('inf')
    x = tl.where(mask, x, neg_inf)

    # Pass 1: block max over valid channels
    block_max = tl.max(x, axis=0)

    # Pass 2: sum of exp(x - block_max)
    x_shift = x - block_max
    exp_x = tl.exp(x_shift)
    sum_exp = tl.sum(exp_x, axis=0)

    # Pass 3: probabilities, then tanh, then scale
    prob = exp_x / sum_exp  # valid lanes only (invalid had x=-inf -> exp=0)

    # tanh(prob) = (e^{2p} - 1) / (e^{2p} + 1)
    e2p = tl.exp(2.0 * prob)
    tanh_prob = (e2p - 1.0) / (e2p + 1.0)

    out = tanh_prob * scaling

    # Store (cast to output dtype assumed float)
    out_ptrs = y_ptr + base + offs_c * (H * W)
    tl.store(out_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version that keeps ConvTranspose3d + mean in PyTorch,
    and fuses (add bias) + softmax (over channels) + tanh + scale into a single Triton kernel.

    Entry point: ModelNew
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Bias shape [1, C, 1, 1, 1]
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # 1) ConvTranspose3d
        x = self.conv_transpose(x)  # [B, C, D, H, W]
        # 2) Mean over depth -> [B, C, 1, H, W]
        x = x.mean(dim=2, keepdim=True)

        # CPU fallback: use original PyTorch ops
        if not x.is_cuda:
            x = x + self.bias
            x = torch.softmax(x, dim=1)
            x = torch.tanh(x)
            x = x * self.scaling_factor
            return x

        # Ensure contiguous and view as [B, C, H, W]
        x = x.contiguous()
        B, C, D, H, W = x.shape
        assert D == 1, f"Expected D=1 after mean, got D={D}"
        x_2d = x.view(B, C, H, W).contiguous()

        # Prepare output
        y = torch.empty_like(x_2d)

        # Bias: squeeze to [C]
        bias = self.bias
        bias_1d = bias.view(-1).contiguous()

        # Cast to float for kernel (compute in f32)
        x_f = x_2d.float()
        bias_f = bias_1d.float()
        y_f = y.float()

        # Launch config
        BLOCK_C = 128
        grid = (B * H * W, triton.cdiv(C, BLOCK_C))

        _fused_add_softmax_tanh_scale_2d_kernel[grid](
            x_f, bias_f, y_f,
            B, C, H, W,
            self.scaling_factor,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )

        # Cast back to original dtype if needed and view shape
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        y = y.view(B, C, 1, H, W)
        return y
