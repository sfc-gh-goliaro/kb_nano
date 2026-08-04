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
    def _depth_mean_3d_kernel(
        x_ptr,            # *f32, shape [B, C, D, H, W]
        out_ptr,          # *f32, shape [B, C, 1, H, W]
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
        out_stride_d: tl.constexpr,   # typically 0 if size-1, but pass real
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        # Grid: (B*H*W,)
        pid = tl.program_id(0)
        num_hw = H * W
        b = pid // num_hw
        rem = pid % num_hw
        h = rem // W
        w = rem % W

        base = b * stride_b + h * stride_h + w * stride_w

        # Accumulator per channel lane
        SUM_c = tl.zeros((BLOCK_C,), dtype=tl.float32)

        # Loop over depth
        for d in range(0, D):
            for c0 in range(0, C, BLOCK_C):
                offs_c = c0 + tl.arange(0, BLOCK_C)
                mask = offs_c < C
                ptr = x_ptr + base + d * stride_d + offs_c * stride_c
                vals = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
                SUM_c += tl.where(mask, vals, 0.0)

        mean_c = SUM_c / D

        # Store to out[b, c, 0, h, w]
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            out_ptrs = out_ptr + (b * out_stride_b
                                  + offs_c * out_stride_c
                                  + 0 * out_stride_d
                                  + h * out_stride_h
                                  + w * out_stride_w)
            tl.store(out_ptrs, mean_c, mask=mask)


    @triton.jit
    def _fused_bias_softmax_tanh_scale_2d_kernel(
        x_ptr,            # *f32, shape [B, C, 1, H, W] (after mean)
        bias_ptr,         # *f32, shape [C] (flattened)
        out_ptr,          # *f32, shape [B, C, H, W] output
        B: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        stride_b: tl.constexpr,
        stride_c: tl.constexpr,
        stride_d: tl.constexpr,   # d=0, but pass real stride
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        scale: tl.float32,
        BLOCK_C: tl.constexpr,
    ):
        # Grid: (B*H*W,)
        pid = tl.program_id(0)
        num_hw = H * W
        b = pid // num_hw
        rem = pid % num_hw
        h = rem // W
        w = rem % W

        base = b * stride_b + h * stride_h + w * stride_w

        # Pass 1: max over channels
        max_v = tl.full((), -float('inf'), dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            v = tl.load(x_ptr + base + offs_c * stride_c, mask=mask, other=0.0).to(tl.float32)
            bvals = tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
            v = v + bvals
            v_masked = tl.where(mask, v, -float('inf'))
            block_max = tl.max(v_masked, axis=0)
            max_v = tl.maximum(max_v, block_max)

        # Pass 2: sum of exp
        sum_exp = tl.zeros((), dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            v = tl.load(x_ptr + base + offs_c * stride_c, mask=mask, other=0.0).to(tl.float32)
            bvals = tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
            v = v + bvals
            e = tl.exp(v - max_v)
            e = tl.where(mask, e, 0.0)
            sum_exp += tl.sum(e, axis=0)

        # Pass 3: write normalized, tanh-scaled results
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = offs_c < C
            v = tl.load(x_ptr + base + offs_c * stride_c, mask=mask, other=0.0).to(tl.float32)
            bvals = tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
            v = v + bvals
            e = tl.exp(v - max_v)
            p = e / sum_exp
            # tanh via sigmoid: tanh(z) = 2*sigmoid(2z) - 1
            # sigmoid(t) = 1 / (1 + exp(-t))
            sig = 1.0 / (1.0 + tl.exp(-2.0 * p * scale))
            t = 2.0 * sig - 1.0
            out_base = b * out_stride_b + h * out_stride_h + w * out_stride_w
            out_ptrs = out_ptr + out_base + offs_c * out_stride_c
            tl.store(out_ptrs, t, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model:
    - Keeps ConvTranspose3d in PyTorch (cuDNN).
    - Replaces mean over depth + bias + softmax + tanh + scale with two Triton kernels:
        1) depth_mean_3d_kernel: mean over D -> (B, C, 1, H, W)
        2) fused_bias_softmax_tanh_scale_2d_kernel: bias+softmax+tanh+scale -> (B, C, H, W)
      Then view as (B, C, 1, H, W).
    - Falls back to PyTorch if Triton/CUDA not available.
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

        # Fallback if Triton not available or not CUDA
        if (not TRITON_AVAILABLE) or (device.type != "cuda"):
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

        # 3) Allocate intermediate mean output (B, C, 1, H, W) float32
        y_mean = torch.empty((B, C, 1, H, W), device=device, dtype=torch.float32)

        # Strides for y_conv (B, C, D, H, W)
        sb, sc, sd, sh, sw = y_conv.stride()
        # Strides for y_mean (B, C, 1, H, W)
        smb, msc, msd, msh, msw = y_mean.stride()

        # Launch depth mean kernel: grid over P = B*H*W
        BLOCK_C = 128 if C >= 128 else 64
        P = B * H * W
        grid = (P,)

        _depth_mean_3d_kernel[grid](
            y_conv, y_mean,
            B, C, D, H, W,
            sb, sc, sd, sh, sw,
            smb, msc, msd, msh, msw,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )

        # 4) Prepare bias as 1D float32
        bias_1d = self.bias.view(-1).to(device=device, dtype=torch.float32)

        # 5) Allocate output (B, C, H, W) float32
        out = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        # Strides for y_mean view as (B, C, 1, H, W) but we'll index as (B, C, H, W) by fixing d=0
        # Use real strides: b, c, h, w
        smb, msc, msd, msh, msw = y_mean.stride()  # msd is stride for size-1 dim
        # But to simplify, view y_mean as (B, C, H, W, 1) and take strides of (B, C, H, W):
        # However, view changes strides; safer to recompute from storage:
        # We'll use y_mean as-is and pass its strides; inside kernel we'll use d=0 offset msd=0 or.ignore d.
        # Easier: make a contiguous view (B, C, H, W) from y_mean by squeeze or view?
        # y_mean.select(2, 0) returns a view (B, C, H, W) with correct strides.
        y_mean_2d = y_mean.select(2, 0)  # (B, C, H, W) view
        ym_sb, ym_sc, ym_sh, ym_sw = y_mean_2d.stride()

        # Out strides
        out_sb, out_sc, out_sh, out_sw = out.stride()

        # Launch fused kernel over P = B*H*W
        _fused_bias_softmax_tanh_scale_2d_kernel[grid](
            y_mean_2d, bias_1d, out,
            B, C, H, W,
            ym_sb, ym_sc, 0, ym_sh, ym_sw,  # stride_d=0 since we use 2D view
            out_sb, out_sc, out_sh, out_sw,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )

        # 6) Reshape to (B, C, 1, H, W)
        out = out.view(B, C, 1, H, W)
        return out
