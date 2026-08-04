import torch
import torch.nn as nn

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_mean_softmax_tanh_scale_3d_kernel(
        x_ptr,            # *float, shape [B, C, D, H, W] (we'll pass contiguous)
        bias_ptr,         # *float, shape [C]
        out_ptr,          # *float, shape [B, C, H, W] (after mean over D)
        B: tl.constexpr,
        C: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        stride_b: tl.constexpr,
        stride_c: tl.constexpr,
        stride_d: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        scale: tl.float32,
        BLOCK_C: tl.constexpr,
    ):
        # Program id: one program per (b, h, w)
        pid = tl.program_id(0)
        num_hw = H * W
        b = pid // num_hw
        rem = pid % num_hw
        h = rem // W
        w = rem % W

        # Base pointer offset for this (b, h, w)
        base = b * stride_b + h * stride_h + w * stride_w

        # 1) Compute mean over depth D: sum_{d} x[b, c, d, h, w] / D
        sum_c = tl.zeros((), dtype=tl.float32)
        for d in range(0, D):
            # accumulate over channels in BLOCK_C chunks
            for c0 in range(0, C, BLOCK_C):
                offs_c = c0 + tl.arange(0, BLOCK_C)
                mask = offs_c < C
                ptr = x_ptr + base + d * stride_d + offs_c * stride_c
                vals = tl.load(ptr, mask=mask, other=0.0)
                vals_f32 = vals.to(tl.float32)
                # zero out invalid lanes
                vals_f32 = tl.where(mask, vals_f32, 0.0)
                sum_c += tl.sum(vals_f32, axis=0)
            # Note: sum_c currently holds sum over this d for all channels; but we want sum over d for each channel.
            # Fix: we need an accumulator per channel. Use a python list of scalars is not allowed; instead, use
            # a BLOCK_C vector accumulator across d loops.
            pass  # see below for corrected implementation

        # The above sketch is incorrect because sum_c was a scalar. We need a per-channel accumulator.
        # Revised plan: use a vector accumulator acc over channels and loop d, then compute mean = acc / D.

    # Redefine correctly with per-channel accumulator
    @triton.jit
    def _fused_mean_softmax_tanh_scale_3d_kernel(
        x_ptr, bias_ptr, out_ptr,
        B: tl.constexpr, C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
        stride_b: tl.constexpr, stride_c: tl.constexpr, stride_d: tl.constexpr,
        stride_h: tl.constexpr, stride_w: tl.constexpr,
        out_stride_b: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
        scale: tl.float32,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_hw = H * W
        b = pid // num_hw
        rem = pid % num_hw
        h = rem // W
        w = rem % W

        base = b * stride_b + h * stride_h + w * stride_w

        # Per-channel accumulator for sum over depth D
        # We'll keep it as a python list of scalars is not supported; use a BLOCK_C vector and reduce later.
        # Easiest: loop d, for each chunk compute sum over channels and accumulate into a BLOCK_C vector SUM_c.
        SUM_c = tl.zeros((BLOCK_C,), dtype=tl.float32)

        # Loop over depth
        for d in range(0, D):
            for c0 in range(0, C, BLOCK_C):
                offs_c = c0 + tl.arange(0, BLOCK_C)
                mask = offs_c < C
                ptr = x_ptr + base + d * stride_d + offs_c * stride_c
                vals = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
                # Accumulate this d's contribution into SUM_c
                SUM_c += tl.where(mask, vals, 0.0)

        # mean per channel = SUM_c / D
        mean_c = SUM_c / D  # shape (BLOCK_C,)

        # Now softmax over channels: need max, then sum exp, then normalize.
        max_v = tl.full((), -float('inf'), dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            v = mean_c + tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
            v_masked = tl.where(mask, v, -float('inf'))
            block_max = tl.max(v_masked, axis=0)
            max_v = tl.maximum(max_v, block_max)

        sum_exp = tl.zeros((), dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            v = mean_c + tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
            e = tl.exp(v - max_v)
            e = tl.where(mask, e, 0.0)
            sum_exp += tl.sum(e, axis=0)

        # Final write: p = exp / sum, t = tanh(p * scale), store
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            v = mean_c + tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
            e = tl.exp(v - max_v)
            p = e / sum_exp
            t = tl.tanh(p * scale)
            out_base = b * out_stride_b + h * out_stride_h + w * out_stride_w
            out_ptrs = out_ptr + out_base + offs_c * out_stride_c
            tl.store(out_ptrs, t, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version:
    - Keeps ConvTranspose3d in PyTorch (cuDNN).
    - Fuses: mean over depth + bias add + softmax (over channels) + tanh + scale into one Triton kernel.
    Entry point must be ModelNew with same forward signature.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Match original API: extra broadcast bias
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # 1) ConvTranspose3d
        y_conv = self.conv_transpose(x)  # (B, C_out, D, H, W)
        device = y_conv.device
        dtype = y_conv.dtype

        # Fallback if Triton not available or not CUDA
        if (not TRITON_AVAILABLE) or (device.type != "cuda"):
            # Pure PyTorch path identical to original
            y = y_conv.mean(dim=2, keepdim=True)   # (B, C, 1, H, W)
            y = y + self.bias                     # add bias
            y = torch.softmax(y, dim=1)           # softmax over channels
            y = torch.tanh(y)                     # tanh
            y = y * self.scaling_factor           # scale
            return y

        # 2) Ensure contiguous
        if not y_conv.is_contiguous():
            y_conv = y_conv.contiguous()

        B, C, D, H, W = y_conv.shape

        # 3) Prepare bias as 1D
        # Bias shape is (1, C, 1, 1, 1); view as (C,)
        bias_1d = self.bias.view(-1).to(device=device, dtype=torch.float32)

        # 4) Allocate output (B, C, H, W); we will return view as (B, C, 1, H, W)
        out = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        # 5) Launch Triton kernel
        # Strides for x (B, C, D, H, W)
        stride_b, stride_c, stride_d, stride_h, stride_w = y_conv.stride()
        # Strides for out (B, C, H, W)
        out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()

        # Grid: one program per (b, h, w)
        P = B * H * W
        BLOCK_C = 128 if C >= 128 else 64
        grid = (P,)

        _fused_mean_softmax_tanh_scale_3d_kernel[grid](
            y_conv, bias_1d, out,
            B, C, D, H, W,
            stride_b, stride_c, stride_d, stride_h, stride_w,
            out_stride_b, out_stride_c, out_stride_h, out_stride_w,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )

        # 6) Reshape to (B, C, 1, H, W)
        out = out.view(B, C, 1, H, W)

        return out
