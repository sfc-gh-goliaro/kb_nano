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
def _fused_add_softmax_tanh_scale_kernel(
    x_ptr,            # *const T, (B, C, 1, H, W) after mean
    bias_ptr,         # *const T, (1, C, 1, 1, 1)
    out_ptr,          # *T, (B, C, 1, H, W)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,  # present for completeness; d=0
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    scale: tl.constexpr,     # python float
    BLOCK_C: tl.constexpr,
):
    # One program per (b, h, w)
    pid = tl.program_id(0)
    HW = H * W
    b = pid // HW
    rem = pid % HW
    h = rem // W
    w = rem % W

    # base offset for this (b, h, w) location, d=0
    base = b * stride_b + h * stride_h + w * stride_w

    # Pass 1: max over channels in float32 (numerical stability)
    max_val = -float("inf")
    c0 = 0
    while c0 < C:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < C
        x_ptrs = x_ptr + base + offs * stride_c  # + 0 * stride_d
        vals = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        vals_f32 = vals.to(tl.float32)
        # add bias
        bvals = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = vals_f32 + bvals
        # masked max reduction over this chunk
        local_max = tl.max(tl.where(mask, v, -float("inf")), axis=0)
        max_val = tl.maximum(max_val, local_max)
        c0 += BLOCK_C

    # Pass 2: sum of exp(v - max)
    sum_exp = 0.0
    c0 = 0
    while c0 < C:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < C
        x_ptrs = x_ptr + base + offs * stride_c
        vals = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        bvals = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        e = vals + bvals - max_val
        ex = tl.exp(e)
        ex = tl.where(mask, ex, 0.0)
        sum_exp += tl.sum(ex, axis=0)
        c0 += BLOCK_C

    # Pass 3: write softmax -> custom tanh -> scale
    c0 = 0
    while c0 < C:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < C
        x_ptrs = x_ptr + base + offs * stride_c
        vals = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        bvals = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        numer = tl.exp(vals + bvals - max_val)
        softmax = numer / sum_exp  # in [0,1]
        # tanh(softmax): use exp-based formula, stable for s in [0,1]
        # tanh(s) = (1 - exp(-2s)) / (1 + exp(-2s))
        z = -2.0 * softmax
        ez = tl.exp(z)
        tanh_s = (1.0 - ez) / (1.0 + ez)
        y = tanh_s * scale
        # store as float32
        out_ptrs = out_ptr + base + offs * stride_c
        tl.store(out_ptrs, y, mask=mask)
        c0 += BLOCK_C


class ModelNew(nn.Module):
    """
    Triton-optimized version:
      - Keep ConvTranspose3d in PyTorch (cuDNN)
      - Fuse mean -> add(bias) -> softmax -> tanh -> scale into a single Triton kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        # Match original: learnable broadcast bias (1, C, 1, 1, 1)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

        if not TRITON_AVAILABLE:
            print("Warning: Triton is not available; ModelNew will fall back to PyTorch ops.")

    def forward(self, x: torch.Tensor):
        """
        x: (B, C_in, D, H, W)
        returns: (B, C_out, 1, H', W')
        """
        # 1) ConvTranspose3d (cuDNN)
        y = self.conv_transpose(x)  # (B, C_out, D', H', W')

        # 2) Mean over depth, keepdim
        y = y.mean(dim=2, keepdim=True)  # (B, C_out, 1, H', W')

        # Fallback if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (not y.is_cuda):
            y = y + self.bias
            y = torch.softmax(y, dim=1)
            y = torch.tanh(y)
            y = y * self.scaling_factor
            return y

        # Ensure contiguous for clean strides
        y = y.contiguous()
        bias = self.bias
        if bias.device != y.device:
            bias = bias.to(y.device)
        if bias.dtype != y.dtype:
            bias = bias.to(y.dtype)

        B, C, D, H, W = y.shape
        assert D == 1, f"Expected D=1 after mean, got D={D}"

        # Output buffer
        out = torch.empty_like(y)

        # Strides in elements
        sb, sc, sd, sh, sw = y.stride()
        # Grid: one program per (b,h,w)
        grid = (B * H * W,)

        # Choose BLOCK_C as next power-of-two >= C, capped to 1024
        block_c = 1
        while block_c < C and block_c < 1024:
            block_c <<= 1
        block_c = max(block_c, 1)

        # Heuristic for num_warps
        num_warps = 4 if block_c <= 128 else 8

        _fused_add_softmax_tanh_scale_kernel[grid](
            y, bias, out,
            B, C, H, W,
            sb, sc, sd, sh, sw,
            self.scaling_factor,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )

        return out
