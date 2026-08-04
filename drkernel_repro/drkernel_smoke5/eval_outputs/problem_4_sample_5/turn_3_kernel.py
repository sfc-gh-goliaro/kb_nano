import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# =============================
# Triton kernels (float32, NCHW)
# =============================

@triton.jit
def conv1x1_pointwise_fused_kernel(
    x_ptr, w_ptr, bconv_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_in, C_out, H, W,
    stride_n, stride_c, stride_h, stride_w,
    w_stride_oc, w_stride_ic,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    has_bn: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (N, tiles_HW, tiles_C)
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)

    num_w_tiles = tl.cdiv(W, BLOCK_W)
    th = pid_hw // num_w_tiles
    tw = pid_hw % num_w_tiles

    h = th * BLOCK_H + tl.arange(0, BLOCK_H)
    w = tw * BLOCK_W + tl.arange(0, BLOCK_W)
    oc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = (h < H) & (w < W)
    mask_oc = oc < C_out

    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Loop over input channels in blocks
    for ic in range(0, C_in, BLOCK_C):
        ic_ = ic + tl.arange(0, BLOCK_C)
        mask_ic = ic_ < C_in

        # Load weights W[oc, ic] -> [BC,BIC]
        w_idx = (oc[:, None] * w_stride_oc) + (ic_[None, :] * w_stride_ic)
        w_val = tl.load(w_ptr + w_idx, mask=(mask_oc[:, None] & mask_ic[None, :]), other=0.0)  # [BC,BIC]

        # Load x[n, ic, h, w] -> [BIC,BH,BW]
        x_idx = (
            pid_n * stride_n
            + ic_[:, None, None] * stride_c
            + h[None, :, None] * stride_h
            + w[None, None, :] * stride_w
        )
        x_mask = (mask_ic[:, None, None]) & (mask_hw[None, :, None])
        x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)  # [BIC,BH,BW]

        # Accumulate: for each bic, acc += x[:,:,bic] * w[bic,:]
        x_t = tl.trans(x_val, (1, 2, 0))  # [BH,BW,BIC]
        w_t = tl.trans(w_val, (1, 0))     # [BIC,BC]
        for bic in range(0, BLOCK_C):
            if bic < C_in - ic:  # simple guard
                x_slice = x_t[:, :, bic]  # [BH,BW]
                w_slice = w_t[bic, :]     # [BC]
                acc += x_slice[:, :, None] * w_slice[None, None, :]

    # Add conv bias
    bconv = tl.load(bconv_ptr + oc, mask=mask_oc, other=0.0)
    acc = acc + bconv[None, None, :]

    if has_bn:
        # BN(eval): scale = weight / sqrt(var + eps); shift = bias - mean * scale
        scale = tl.load(weight_ptr + oc, mask=mask_oc, other=1.0) / tl.sqrt(tl.load(bias_ptr + oc, mask=mask_oc, other=1.0) + eps)
        shift = tl.load(bias_ptr + oc, mask=mask_oc, other=0.0) - tl.load(weight_ptr + oc, mask=mask_oc, other=1.0) * scale
        acc = acc * scale[None, None, :] + shift[None, None, :]

    # ReLU6
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    # Store
    out_idx = (
        pid_n * out_stride_n
        + oc[:, None, None] * out_stride_c
        + h[None, :, None] * out_stride_h
        + w[None, None, :] * out_stride_w
    )
    out_mask = (mask_oc[:, None, None]) & (mask_hw[None, :, None])
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def dw_conv_kxk_fused_kernel(
    x_ptr, w_ptr, bias_ptr, weight_ptr, bias2_ptr, out_ptr,
    N, C, H, W,
    K: tl.constexpr,
    stride_n, stride_c, stride_h, stride_w,
    w_stride_c, w_stride_dh, w_stride_dw,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    has_bn: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (N, tiles_HO*WO, tiles_C)
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)

    num_w_tiles = tl.cdiv(W, BLOCK_W)
    th = pid_hw // num_w_tiles
    tw = pid_hw % num_w_tiles

    oh = th * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = tw * BLOCK_W + tl.arange(0, BLOCK_W)
    c  = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    HO, WO = H, W  # no pooling; output size equals input
    PH = (H - 1) // 2
    PW = (W - 1) // 2

    mask_hw = (oh < HO) & (ow < WO)
    mask_c  = c < C

    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Loop over kernel KxK
    for dh in range(0, K):
        for dw in range(0, K):
            ih = oh + dh - PH
            iw = ow + dw - PW
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask_hw
            # x[n, c, ih, iw] -> [BC,BH,BW]
            x_idx = (
                pid_n * stride_n
                + c[:, None, None] * stride_c
                + ih[None, :, None] * stride_h
                + iw[None, None, :] * stride_w
            )
            x_val = tl.load(x_ptr + x_idx, mask=(mask_c[:, None, None] & valid[None, :, None]), other=0.0)  # [BC,BH,BW]

            # w[c, dh, dw] -> [BC]
            w_idx = c * w_stride_c + dh * w_stride_dh + dw * w_stride_dw
            w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)  # [BC]

            acc += x_val * w_val[:, None, None]

    # Bias (pre-activation)
    b = tl.load(bias_ptr + c, mask=mask_c, other=0.0)
    acc = acc + b[None, None, :]

    if has_bn:
        scale = tl.load(weight_ptr + c, mask=mask_c, other=1.0) / tl.sqrt(tl.load(bias2_ptr + c, mask=mask_c, other=1.0) + eps)
        shift = tl.load(bias2_ptr + c, mask=mask_c, other=0.0) - tl.load(weight_ptr + c, mask=mask_c, other=1.0) * scale
        acc = acc * scale[None, None, :] + shift[None, None, :]

    # ReLU6
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    # Store
    out_idx = (
        pid_n * out_stride_n
        + c[:, None, None] * out_stride_c
        + oh[None, :, None] * out_stride_h
        + ow[None, None, :] * out_stride_w
    )
    out_mask = (mask_c[:, None, None]) & (mask_hw[None, :, None])
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def conv1x1_project_fused_kernel(
    x_ptr, w_ptr, bconv_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_in, C_out, H, W,
    stride_n, stride_c, stride_h, stride_w,
    w_stride_oc, w_stride_ic,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    has_bn: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Same structure as expand but from C_in->C_out
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)

    num_w_tiles = tl.cdiv(W, BLOCK_W)
    th = pid_hw // num_w_tiles
    tw = pid_hw % num_w_tiles

    h = th * BLOCK_H + tl.arange(0, BLOCK_H)
    w = tw * BLOCK_W + tl.arange(0, BLOCK_W)
    oc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = (h < H) & (w < W)
    mask_oc = oc < C_out

    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    for ic in range(0, C_in, BLOCK_C):
        ic_ = ic + tl.arange(0, BLOCK_C)
        mask_ic = ic_ < C_in

        w_idx = (oc[:, None] * w_stride_oc) + (ic_[None, :] * w_stride_ic)
        w_val = tl.load(w_ptr + w_idx, mask=(mask_oc[:, None] & mask_ic[None, :]), other=0.0)  # [BC,BIC]

        x_idx = (
            pid_n * stride_n
            + ic_[:, None, None] * stride_c
            + h[None, :, None] * stride_h
            + w[None, None, :] * stride_w
        )
        x_mask = (mask_ic[:, None, None]) & (mask_hw[None, :, None])
        x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)  # [BIC,BH,BW]

        x_t = tl.trans(x_val, (1, 2, 0))  # [BH,BW,BIC]
        w_t = tl.trans(w_val, (1, 0))     # [BIC,BC]
        for bic in range(0, BLOCK_C):
            if bic < C_in - ic:
                x_slice = x_t[:, :, bic]  # [BH,BW]
                w_slice = w_t[bic, :]     # [BC]
                acc += x_slice[:, :, None] * w_slice[None, None, :]

    bconv = tl.load(bconv_ptr + oc, mask=mask_oc, other=0.0)
    acc = acc + bconv[None, None, :]

    if has_bn:
        scale = tl.load(weight_ptr + oc, mask=mask_oc, other=1.0) / tl.sqrt(tl.load(bias_ptr + oc, mask=mask_oc, other=1.0) + eps)
        shift = tl.load(bias_ptr + oc, mask=mask_oc, other=0.0) - tl.load(weight_ptr + oc, mask=mask_oc, other=1.0) * scale
        acc = acc * scale[None, None, :] + shift[None, None, :]

    # No activation on projection

    out_idx = (
        pid_n * out_stride_n
        + oc[:, None, None] * out_stride_c
        + h[None, :, None] * out_stride_h
        + w[None, None, :] * out_stride_w
    )
    out_mask = (mask_oc[:, None, None]) & (mask_hw[None, :, None])
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# =============================
# Python wrappers
# =============================

def _launch_conv1x1_pointwise(x, w, bconv, weight, bias, has_bn, eps,
                              BLOCK_C=64, BLOCK_H=32, BLOCK_W=32):
    assert x.is_cuda and w.is_cuda and bconv.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32 and bconv.dtype == torch.float32
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    out = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_h, stride_w = x.stride()
    w_stride_oc = w.stride(0)
    w_stride_ic = w.stride(1)

    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    grid = (N, triton.cdiv(H, BLOCK_H) * triton.cdiv(W, BLOCK_W), triton.cdiv(C_out, BLOCK_C))

    conv1x1_pointwise_fused_kernel[grid](
        x, w, bconv,
        weight, bias,
        out,
        N, C_in, C_out, H, W,
        stride_n, stride_c, stride_h, stride_w,
        w_stride_oc, w_stride_ic,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        has_bn=has_bn,
        eps=eps,
        BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4,
    )
    return out


def _launch_dw_conv_kxk(x, w, bias, weight, bias2, K, has_bn, eps,
                        BLOCK_C=64, BLOCK_H=32, BLOCK_W=32):
    assert x.is_cuda and w.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32 and bias.dtype == torch.float32
    N, C, H, W = x.shape
    assert w.shape[0] == C and w.shape[1] == K and w.shape[2] == K
    out = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_h, stride_w = x.stride()
    w_stride_c, w_stride_dh, w_stride_dw = w.stride()

    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    grid = (N, triton.cdiv(H, BLOCK_H) * triton.cdiv(W, BLOCK_W), triton.cdiv(C, BLOCK_C))

    dw_conv_kxk_fused_kernel[grid](
        x, w, bias,
        weight, bias2,
        out,
        N, C, H, W,
        K,
        stride_n, stride_c, stride_h, stride_w,
        w_stride_c, w_stride_dh, w_stride_dw,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        has_bn=has_bn,
        eps=eps,
        BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4,
    )
    return out


def _launch_conv1x1_project(x, w, bconv, weight, bias, has_bn, eps,
                            BLOCK_C=64, BLOCK_H=32, BLOCK_W=32):
    assert x.is_cuda and w.is_cuda and bconv.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32 and bconv.dtype == torch.float32
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    out = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_h, stride_w = x.stride()
    w_stride_oc = w.stride(0)
    w_stride_ic = w.stride(1)

    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    grid = (N, triton.cdiv(H, BLOCK_H) * triton.cdiv(W, BLOCK_W), triton.cdiv(C_out, BLOCK_C))

    conv1x1_project_fused_kernel[grid](
        x, w, bconv,
        weight, bias,
        out,
        N, C_in, C_out, H, W,
        stride_n, stride_c, stride_h, stride_w,
        w_stride_oc, w_stride_ic,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        has_bn=has_bn,
        eps=eps,
        BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4,
    )
    return out


# =============================
# Entry point: ModelNew (Triton)
# =============================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.expand_bn   = nn.BatchNorm2d(hidden_dim)
        else:
            self.expand_conv = None
            self.expand_bn   = None

        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=True)
        self.depthwise_bn   = nn.BatchNorm2d(hidden_dim)

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.project_bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        # Fallback to pure PyTorch if:
        # - Triton not available
        # - tensor not on CUDA
        # - or in training mode
        if (not TRITON_AVAILABLE) or (not x.is_cuda) or self.training:
            identity = x
            if self.expand_conv is not None:
                x = self.expand_conv(x)
                if self.expand_bn is not None:
                    x = self.expand_bn(x)
                x = F.relu6(x)
            x = self.depthwise_conv(x)
            if self.depthwise_bn is not None:
                x = self.depthwise_bn(x)
            x = F.relu6(x)
            x = self.project_conv(x)
            if self.project_bn is not None:
                x = self.project_bn(x)
            if self.use_residual:
                x = x + identity
            return x

        # Triton path (eval, CUDA)
        identity = x
        device = x.device
        assert x.dtype == torch.float32, "This Triton path currently supports float32 only."
        eps = 1e-5

        # 1) Expand 1x1 (if any)
        if self.expand_conv is not None:
            w = self.expand_conv.weight
            b = self.expand_conv.bias
            running_mean = self.expand_bn.running_mean
            running_var  = self.expand_bn.running_var
            weight_bn    = self.expand_bn.weight
            bias_bn      = self.expand_bn.bias

            has_bn = True
            x = _launch_conv1x1_pointwise(
                x, w, b,
                weight_bn, bias_bn,
                has_bn=True, eps=eps
            )
        else:
            # No expand: x stays as is
            pass

        # 2) Depthwise KxK
        w_d = self.depthwise_conv.weight
        b_d = self.depthwise_conv.bias
        running_mean_d = self.depthwise_bn.running_mean
        running_var_d  = self.depthwise_bn.running_var
        weight_bn_d    = self.depthwise_bn.weight
        bias_bn_d      = self.depthwise_bn.bias

        has_bn_d = True
        x = _launch_dw_conv_kxk(
            x, w_d, b_d,
            weight_bn_d, bias_bn_d,
            K=self.depthwise_conv.kernel_size[0],
            has_bn=has_bn_d, eps=eps
        )
        # Activation after depthwise in original: ReLU6
        x = F.relu6(x)

        # 3) Project 1x1
        w_p = self.project_conv.weight
        b_p = self.project_conv.bias
        running_mean_p = self.project_bn.running_mean
        running_var_p  = self.project_bn.running_var
        weight_bn_p    = self.project_bn.weight
        bias_bn_p      = self.project_bn.bias

        has_bn_p = True
        out = _launch_conv1x1_project(
            x, w_p, b_p,
            weight_bn_p, bias_bn_p,
            has_bn=has_bn_p, eps=eps
        )

        # 4) Residual add if needed
        if self.use_residual:
            out = out + identity

        return out


# Keep Model definition for compatibility (not strictly required by the task, but helpful)
class Model(ModelNew):
    pass
