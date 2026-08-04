import math
import torch
import torch.nn as nn

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# ---------------------------
# Kernel: 1x1 Conv + BN + ReLU6
# x: [N, C_in, H, W], w: [C_out, C_in, 1, 1]
# bn: weight [C_out], bias [C_out], running_mean [C_out], running_var [C_out], eps scalar
# y: [N, C_out, H, W]
# ---------------------------
@triton.jit
def conv1x1_bn_relu6_kernel(
    x_ptr, w_ptr, y_ptr,
    bn_weight_ptr, bn_bias_ptr, bn_mean_ptr, bn_var_ptr,
    N: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr,
    stride_n: tl.constexpr, stride_c: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    w_stride_co: tl.constexpr, w_stride_ci: tl.constexpr,  # strides for w (no kh/kw since 1x1)
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_PX: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    # Program ids
    pid_px = tl.program_id(0)  # over flattened pixels
    pid_oc = tl.program_id(1)   # over output channels

    # Offsets
    px = pid_px * BLOCK_PX + tl.arange(0, BLOCK_PX)
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    # Bounds
    total_px = N * H * W
    mask_px = px < total_px
    mask_oc = oc < C_out

    # Decode (n, h, w) from px
    HW = H * W
    n = px // HW
    rem = px % HW
    h = rem // W
    w = rem % W

    # Accumulator
    acc = tl.zeros((BLOCK_PX, BLOCK_OC), dtype=tl.float32)

    # Loop over input channels in blocks
    for ic0 in range(0, C_in, BLOCK_IC):
        # Loop over i within the block; simple scalar loop to avoid advanced indexing
        for i in range(0, BLOCK_IC):
            ic_i = ic0 + i
            valid_i = ic_i < C_in

            # Load X vector: x[n, ic_i, h, w] -> shape (PX,)
            x_off = n * stride_n + ic_i * stride_c + h * stride_h + w * stride_w
            x_vals = tl.load(x_ptr + x_off, mask=mask_px & valid_i, other=0.0)

            # Load W vector: w[oc, ic_i, 0, 0] -> shape (OC,)
            w_off = oc * w_stride_co + ic_i * w_stride_ci
            w_vals = tl.load(w_ptr + w_off, mask=mask_oc & valid_i, other=0.0)

            # Accumulate: acc += x[:, None] * w[None, :]
            acc += x_vals[:, None] * w_vals[None, :]

    # Apply BN: y = y * weight / sqrt(var + eps) + bias
    bn_w  = tl.load(bn_weight_ptr + oc, mask=mask_oc, other=1.0)       # (OC,)
    bn_b  = tl.load(bn_bias_ptr   + oc, mask=mask_oc, other=0.0)       # (OC,)
    bn_mean = tl.load(bn_mean_ptr + oc, mask=mask_oc, other=0.0)       # (OC,)
    bn_var  = tl.load(bn_var_ptr  + oc, mask=mask_oc, other=1.0)       # (OC,)
    inv_std = 1.0 / tl.sqrt(bn_var + eps)                               # (OC,)

    # Broadcast to (PX, OC)
    acc = acc * (bn_w[None, :] * inv_std[None, :]) + bn_b[None, :]

    # Apply ReLU6
    zero = 0.0
    six = 6.0
    acc = tl.maximum(acc, zero)
    acc = tl.minimum(acc, six)

    # Store
    y_off = n * out_stride_n + oc[None, :] * out_stride_c + h * out_stride_h + w * out_stride_w
    y_mask = mask_px[:, None] & mask_oc[None, :]
    tl.store(y_ptr + y_off, acc, mask=y_mask)


# ---------------------------
# Kernel: Depthwise kxk Conv + BN + ReLU6
# x: [N, C_in, H, W], w: [C_in, 1, 1, Kh, Kw] but we'll index as flattened [C_in*Kh*Kw]
# bn: weight [C_in], bias [C_in], running_mean [C_in], running_var [C_in], eps scalar
# y: [N, C_in, H, W]
# Note: output channels == input channels (depthwise)
# ---------------------------
@triton.jit
def dw_convk_bn_relu6_kernel(
    x_ptr, w_ptr, y_ptr,
    bn_weight_ptr, bn_bias_ptr, bn_mean_ptr, bn_var_ptr,
    N: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Kh: tl.constexpr, Kw: tl.constexpr,
    stride_n: tl.constexpr, stride_c: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_PX: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_px = tl.program_id(0)
    pid_c  = tl.program_id(1)

    px = pid_px * BLOCK_PX + tl.arange(0, BLOCK_PX)
    c  = pid_c  * BLOCK_C  + tl.arange(0, BLOCK_C)

    total_px = N * H * W
    mask_px = px < total_px
    mask_c  = c  < C_in

    HW = H * W
    n  = px // HW
    rem = px % HW
    h  = rem // W
    w  = rem % W

    acc = tl.zeros((BLOCK_PX, BLOCK_C), dtype=tl.float32)

    # Loop over kernel window
    for kh in range(0, Kh):
        for kw in range(0, Kw):
            hi = h + kh
            wi = w + kw
            valid = (hi < H) & (wi < W)

            # x_off = n*stride_n + c*stride_c + hi*stride_h + wi*stride_w
            x_off = n * stride_n + c[None, :] * stride_c + hi[:, None] * stride_h + wi[:, None] * stride_w
            x_mask = mask_px[:, None] & mask_c[None, :] & valid[:, None]
            x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # (PX, C)

            # w index = c * (Kh*Kw) + (kh*Kw + kw)
            idx_w = c * (Kh * Kw) + (kh * Kw + kw)
            w_vals = tl.load(w_ptr + idx_w, mask=mask_c, other=0.0)  # (C,)

            prod = x_vals * w_vals[None, :]  # (PX, C)
            acc += prod

    # Apply BN over input channel c
    bn_w  = tl.load(bn_weight_ptr + c, mask=mask_c, other=1.0)       # (C,)
    bn_b  = tl.load(bn_bias_ptr   + c, mask=mask_c, other=0.0)       # (C,)
    bn_mean = tl.load(bn_mean_ptr + c, mask=mask_c, other=0.0)       # (C,)
    bn_var  = tl.load(bn_var_ptr  + c, mask=mask_c, other=1.0)       # (C,)
    inv_std = 1.0 / tl.sqrt(bn_var + eps)                             # (C,)

    acc = acc * (bn_w[None, :] * inv_std[None, :]) + bn_b[None, :]

    # ReLU6
    zero = 0.0
    six = 6.0
    acc = tl.maximum(acc, zero)
    acc = tl.minimum(acc, six)

    # Store
    y_off = n * out_stride_n + c[None, :] * out_stride_c + h * out_stride_h + w * out_stride_w
    y_mask = mask_px[:, None] & mask_c[None, :]
    tl.store(y_ptr + y_off, acc, mask=y_mask)


# ---------------------------
# Kernel: 1x1 Conv + BN (no activation)
# x: [N, C_in, H, W], w: [C_out, C_in, 1, 1]
# bn: weight [C_out], bias [C_out], running_mean [C_out], running_var [C_out], eps scalar
# y: [N, C_out, H, W]
# ---------------------------
@triton.jit
def conv1x1_bn_kernel(
    x_ptr, w_ptr, y_ptr,
    bn_weight_ptr, bn_bias_ptr, bn_mean_ptr, bn_var_ptr,
    N: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr,
    stride_n: tl.constexpr, stride_c: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    w_stride_co: tl.constexpr, w_stride_ci: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_PX: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_px = tl.program_id(0)
    pid_oc = tl.program_id(1)

    px = pid_px * BLOCK_PX + tl.arange(0, BLOCK_PX)
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    total_px = N * H * W
    mask_px = px < total_px
    mask_oc = oc < C_out

    HW = H * W
    n = px // HW
    rem = px % HW
    h = rem // W
    w = rem % W

    acc = tl.zeros((BLOCK_PX, BLOCK_OC), dtype=tl.float32)

    for ic0 in range(0, C_in, BLOCK_IC):
        for i in range(0, BLOCK_IC):
            ic_i = ic0 + i
            valid_i = ic_i < C_in

            x_off = n * stride_n + ic_i * stride_c + h * stride_h + w * stride_w
            x_vals = tl.load(x_ptr + x_off, mask=mask_px & valid_i, other=0.0)

            w_off = oc * w_stride_co + ic_i * w_stride_ci
            w_vals = tl.load(w_ptr + w_off, mask=mask_oc & valid_i, other=0.0)

            acc += x_vals[:, None] * w_vals[None, :]

    # BN
    bn_w  = tl.load(bn_weight_ptr + oc, mask=mask_oc, other=1.0)
    bn_b  = tl.load(bn_bias_ptr   + oc, mask=mask_oc, other=0.0)
    bn_mean = tl.load(bn_mean_ptr + oc, mask=mask_oc, other=0.0)
    bn_var  = tl.load(bn_var_ptr  + oc, mask=mask_oc, other=1.0)
    inv_std = 1.0 / tl.sqrt(bn_var + eps)

    acc = acc * (bn_w[None, :] * inv_std[None, :]) + bn_b[None, :]

    # Store
    y_off = n * out_stride_n + oc[None, :] * out_stride_c + h * out_stride_h + w * out_stride_w
    y_mask = mask_px[:, None] & mask_oc[None, :]
    tl.store(y_ptr + y_off, acc, mask=y_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            # Keep modules to hold weights & BN params; we'll use their parameters in Triton
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
        else:
            self.expand_conv = None

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Cache kernel sizes
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.hidden_dim = hidden_dim

        # Tuning parameters
        self.BLOCK_PX = 256
        self.BLOCK_OC = 64
        self.BLOCK_IC = 32
        self.num_warps = 4
        self.num_stages = 2

    def forward(self, x: torch.Tensor):
        # Fallback to PyTorch if not CUDA or Triton not available
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            # Original path: expand -> depthwise -> project -> residual
            identity = x
            if self.expand_conv is not None:
                x = self.expand_conv[0](x)
                x = self.expand_conv[1](x)
                x = torch.nn.functional.relu6(x)
            x = self.depthwise_conv[0](x)
            x = self.depthwise_conv[1](x)
            x = torch.nn.functional.relu6(x)
            x = self.project_conv[0](x)
            x = self.project_conv[1](x)
            if self.use_residual:
                x = x + identity
            return x

        # Triton path
        N, C, H, W = x.shape
        device = x.device

        # Ensure float32 compute
        x_f = x.float()

        # 1) Expand: 1x1 conv + BN + ReLU6
        if self.expand_conv is not None:
            w1 = self.expand_conv[0].weight
            bn1 = self.expand_conv[1]
            C_in = C
            C_out = self.hidden_dim
            Kh, Kw = 1, 1

            y1 = torch.empty((N, C_out, H, W), device=device, dtype=torch.float32)

            grid = (triton.cdiv(N * H * W, self.BLOCK_PX), triton.cdiv(C_out, self.BLOCK_OC))
            conv1x1_bn_relu6_kernel[grid](
                x_f, w1, y1,
                bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var,
                N, C_in, H, W, C_out,
                x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
                w1.stride(0), w1.stride(1),
                y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
                bn1.eps,
                BLOCK_PX=self.BLOCK_PX, BLOCK_OC=self.BLOCK_OC, BLOCK_IC=self.BLOCK_IC,
                num_warps=self.num_warps, num_stages=self.num_stages,
            )
        else:
            y1 = x_f  # identity when expand_ratio == 1

        # 2) Depthwise: kxk conv + BN + ReLU6
        dw = self.depthwise_conv[0]
        bndw = self.depthwise_conv[1]
        C_mid = self.hidden_dim
        Kh, Kw = dw.kernel_size
        assert dw.groups == C_mid, "Only depthwise (groups == C) is supported"

        y2 = torch.empty((N, C_mid, H, W), device=device, dtype=torch.float32)

        grid = (triton.cdiv(N * H * W, self.BLOCK_PX), triton.cdiv(C_mid, 64))
        BLOCK_C = 64 if C_mid >= 64 else 32
        dw_convk_bn_relu6_kernel[grid](
            y1, dw.weight.view(-1), y2,
            bndw.weight, bndw.bias, bndw.running_mean, bndw.running_var,
            N, C_mid, H, W,
            Kh, Kw,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            bndw.eps,
            BLOCK_PX=self.BLOCK_PX, BLOCK_C=BLOCK_C,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

        # 3) Project: 1x1 conv + BN (no activation) + residual add
        wp = self.project_conv[0].weight
        bnp = self.project_conv[1]
        C_out = self.out_channels

        y3 = torch.empty((N, C_out, H, W), device=device, dtype=torch.float32)

        grid = (triton.cdiv(N * H * W, self.BLOCK_PX), triton.cdiv(C_out, self.BLOCK_OC))
        conv1x1_bn_kernel[grid](
            y2, wp, y3,
            bnp.weight, bnp.bias, bnp.running_mean, bnp.running_var,
            N, C_mid, H, W, C_out,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            wp.stride(0), wp.stride(1),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            bnp.eps,
            BLOCK_PX=self.BLOCK_PX, BLOCK_OC=self.BLOCK_OC, BLOCK_IC=self.BLOCK_IC,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

        # Residual add if needed
        if self.use_residual:
            identity_f = x.float()
            y3 = y3 + identity_f

        # Cast back to input dtype if needed
        if x.dtype != torch.float32:
            y3 = y3.to(x.dtype)

        return y3
