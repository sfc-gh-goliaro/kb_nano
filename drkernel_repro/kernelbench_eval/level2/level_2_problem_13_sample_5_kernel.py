import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def mean_add_softmax_kernel(
    x_ptr,                 # *f32, shape [B, C, D, H, W] (contiguous or strided)
    bias_ptr,              # *f32, shape [C]
    out_ptr,               # *f32, shape [B, C, H, W] (view as [B, C, 1, H, W])
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
    BLOCK_C: tl.constexpr,
):
    # program ids
    pid_pix = tl.program_id(0)  # over B*H*W
    pid_cb  = tl.program_id(1)  # over channel blocks

    # decode pixel id into (b, h, w)
    HW = H * W
    b = pid_pix // HW
    rem = pid_pix % HW
    h = rem // W
    w = rem % W

    # channel block start
    c0 = pid_cb * BLOCK_C

    # base offset for (b, h, w) using only width/height strides
    base = b * stride_b + h * stride_h + w * stride_w

    # -----------------------------
    # Pass 1: compute global max and sum (two-pass log-sum-exp merge)
    m = -1.0e20
    s = 0.0
    c = c0
    while c < C:
        offs_c = c + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # mean over depth D
        sum_val = tl.zeros([BLOCK_C], dtype=tl.float32)
        d = 0
        while d < D:
            ptr = x_ptr + base + d * stride_d + offs_c * stride_c
            vals = tl.load(ptr, mask=mask_c, other=0.0)
            sum_val += vals
            d += 1
        mean = sum_val / float(D)

        # add bias
        bias_vals = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        z = mean + bias_vals

        # block max with masking
        z_masked = tl.where(mask_c, z, -1.0e20)
        m_b = tl.max(z_masked, axis=0)

        # block sum of exp(z - m_b)
        exp_vals = tl.exp(z - m_b)
        exp_masked = tl.where(mask_c, exp_vals, 0.0)
        s_b = tl.sum(exp_masked, axis=0)

        # merge (m, s) with (m_b, s_b)
        new_m = tl.maximum(m, m_b)
        s = s * tl.exp(m - new_m) + s_b * tl.exp(m_b - new_m)
        m = new_m

        c += BLOCK_C

    # -----------------------------
    # Pass 2: recompute z and write softmax = exp(z - m) / s
    c = c0
    while c < C:
        offs_c = c + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # mean over depth D
        sum_val = tl.zeros([BLOCK_C], dtype=tl.float32)
        d = 0
        while d < D:
            ptr = x_ptr + base + d * stride_d + offs_c * stride_c
            vals = tl.load(ptr, mask=mask_c, other=0.0)
            sum_val += vals
            d += 1
        mean = sum_val / float(D)

        # add bias
        bias_vals = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        z = mean + bias_vals

        soft = tl.exp(z - m) / s

        out_ptrs = out_ptr + b * out_stride_b + offs_c * out_stride_c + h * out_stride_h + w * out_stride_w
        tl.store(out_ptrs, soft, mask=mask_c)

        c += BLOCK_C


@triton.jit
def tanh_scale_kernel(
    in_ptr,                # *f32, shape [B, C, H, W]
    out_ptr,               # *f32, shape [B, C, H, W]
    scale,                 # f32 scalar
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_pix = tl.program_id(0)  # over B*H*W
    pid_cb  = tl.program_id(1)  # over channel blocks

    HW = H * W
    b = pid_pix // HW
    rem = pid_pix % HW
    h = rem // W
    w = rem % W

    c0 = pid_cb * BLOCK_C
    c  = c0 + tl.arange(0, BLOCK_C)
    mask_c = c < C

    base = b * stride_b + h * stride_h + w * stride_w

    in_ptrs  = in_ptr  + base + c * stride_c
    out_ptrs = out_ptr + base + c * stride_c

    x = tl.load(in_ptrs, mask=mask_c, other=0.0)
    t = tl.tanh(x)
    y = t * scale
    tl.store(out_ptrs, y, mask=mask_c)


class ModelNew(nn.Module):
    """
    Triton-optimized version:
    - ConvTranspose3d via cuDNN
    - Fuse mean(D) + bias add + softmax(C) into one Triton kernel (2 passes over channels)
    - Tanh + scale into a second kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))  # broadcast over (B,1,H,W)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # 1) ConvTranspose3d
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors.")
        if x.dtype != torch.float32:
            x = x.float()

        x = x.contiguous()
        B, C, D, H, W = x.shape

        # Create as_strided views to only need width/height strides
        # We will use base = b*sb + h*sh + w*sw, and c*sc, d*sd.
        sb, sc, sd, sh, sw = x.stride()
        x_view = x.as_strided(size=(B, C, D, H, W),
                              stride=(sb, sc, sd, sh, sw))  # no change, but ok to query

        # 2) Allocate intermediate after mean+softmax: (B, C, H, W)
        out1 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

        ob, oc, _, oh, ow = out1.stride()
        out1_view = out1.as_strided(size=(B, C, H, W), stride=(ob, oc, oh, ow))

        # Choose block size and grid
        BLOCK_C = 128
        grid = (B * H * W, triton.cdiv(C, BLOCK_C))

        # Launch mean+add+softmax kernel
        mean_add_softmax_kernel[grid](
            x_view, self.bias.view(-1).contiguous(), out1_view,
            B, C, D, H, W,
            sb, sc, sd, sh, sw,
            ob, oc, oh, ow,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )

        # 3) Tanh + scale
        out2 = torch.empty_like(out1)
        ib, ic, _, ih, iw = out1.stride()
        ob2, oc2, _, oh2, ow2 = out2.stride()

        out1_view2 = out1.as_strided(size=(B, C, H, W), stride=(ib, ic, ih, iw))
        out2_view  = out2.as_strided(size=(B, C, H, W), stride=(ob2, oc2, oh2, ow2))

        tanh_scale_kernel[grid](
            out1_view2, out2_view,
            self.scaling_factor,
            B, C, H, W,
            ib, ic, ih, iw,
            ob2, oc2, oh2, ow2,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )

        # Reshape to (B, C, 1, H, W) to match original API
        return out2.view(B, C, 1, H, W)
