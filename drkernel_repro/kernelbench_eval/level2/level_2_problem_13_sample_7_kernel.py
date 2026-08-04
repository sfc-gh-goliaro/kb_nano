import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _fused_softmax_tanh_scale_5d_kernel(
    x_ptr,           # *T, shape [B, C, 1, H, W]
    bias_ptr,        # *float32, shape [C]
    out_ptr,         # *T, shape [B, C, 1, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,  # D=1
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    scale: tl.constexpr,     # python float
    BLOCK_C: tl.constexpr,
):
    # one program per (b, h, w)
    pid = tl.program_id(0)
    WH = W * H
    b = pid // WH
    rem = pid % WH
    h = rem // W
    w = rem % W

    # base offset for d = 0
    base = b * stride_b + h * stride_h + w * stride_w

    # pass 1: max over channels (in float32)
    m = -float('inf')
    c0 = 0
    while c0 < C:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < C
        ptrs = x_ptr + base + offs * stride_c  # d=0
        v = tl.load(ptrs, mask=mask, other=-float('inf'))
        v32 = v.to(tl.float32)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        v32 = v32 + bias
        v32_masked = tl.where(mask, v32, -float('inf'))
        local_max = tl.max(v32_masked, axis=0)
        m = tl.maximum(m, local_max)
        c0 += BLOCK_C

    # pass 2: sum of exp(v - m) over channels
    s = 0.0
    c0 = 0
    while c0 < C:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < C
        ptrs = x_ptr + base + offs * stride_c
        v = tl.load(ptrs, mask=mask, other=-float('inf'))
        v32 = v.to(tl.float32)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        v32 = v32 + bias
        e = tl.exp(v32 - m)
        e = tl.where(mask, e, 0.0)
        s += tl.sum(e, axis=0)
        c0 += BLOCK_C

    # pass 3: write output = scale * tanh( softmax(v) )
    c0 = 0
    while c0 < C:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < C
        ptrs_in = x_ptr + base + offs * stride_c
        v = tl.load(ptrs_in, mask=mask, other=-float('inf'))
        v32 = v.to(tl.float32)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        v32 = v32 + bias
        num = tl.exp(v32 - m)
        num = tl.where(mask, num, 0.0)
        soft = num / s
        # tanh implementation without tl.tanh:
        # tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
        t = tl.exp(-2.0 * soft)
        tanh_soft = (1.0 - t) / (1.0 + t)
        out32 = scale * tanh_soft
        out = out32.to(v.dtype)
        ptrs_out = out_ptr + base + offs * stride_c
        tl.store(ptrs_out, out, mask=mask)
        c0 += BLOCK_C


class ModelNew(nn.Module):
    """
    Optimized version:
    - ConvTranspose3d kept (cuDNN).
    - Mean over depth kept in PyTorch.
    - Fuses: bias add + softmax over channels + tanh + scale into one Triton kernel.
    Entry point: ModelNew
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Bias shape matches original: (1, C, 1, 1, 1)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor):
        # 1) ConvTranspose3d
        x = self.conv_transpose(x)  # (B, C, D, H, W)

        # 2) Mean over depth, keepdim -> (B, C, 1, H, W)
        x = x.mean(dim=2, keepdim=True)

        # Fallback if no Triton/CUDA
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            x = x + self.bias
            x = torch.softmax(x, dim=1)
            x = torch.tanh(x)
            x = x * self.scaling_factor
            return x

        # Ensure contiguous
        x = x.contiguous()

        # Shapes and strides (element strides)
        assert x.dim() == 5, f"Expected 5D tensor, got {tuple(x.shape)}"
        B, C, D, H, W = x.shape
        # Note: D == 1 due to keepdim after mean
        stride_b, stride_c, stride_d, stride_h, stride_w = x.stride()

        # Allocate output
        out = torch.empty_like(x)

        # Choose BLOCK_C as next power-of-two >= C, capped
        block_c = 1
        while block_c < C:
            block_c <<= 1
        block_c = min(block_c, 1024)

        # Grid: one program per (b, h, w)
        grid = (B * H * W,)

        # Launch kernel
        _fused_softmax_tanh_scale_5d_kernel[grid](
            x, self.bias.view(-1).contiguous(), out,
            B, C, H, W,
            stride_b, stride_c, stride_d, stride_h, stride_w,
            self.scaling_factor,
            BLOCK_C=block_c,
            num_warps=4,
            num_stages=2,
        )

        return out
