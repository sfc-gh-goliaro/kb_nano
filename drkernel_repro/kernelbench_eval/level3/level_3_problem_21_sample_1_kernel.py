import math
import torch
import torch.nn as nn

# Try to import Triton; if unavailable, we'll fallback to PyTorch ops
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def _ceil_div(a, b):
    return (a + b - 1) // b


# Kernel: 1x1 Conv2d (GEMM) + ReLU6
# Computes: y[b, oc, h, w] = relu6( sum_ic x[b, ic, h, w] * w[oc, ic] )
@triton.jit
def expand_1x1_relu6_kernel(
    x_ptr,         # *f32, [B, Cin, H, W]
    w_ptr,         # *f32, [Hidden, Cin]
    y_ptr,         # *f32, [B, Hidden, H, W]
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,   # Hidden
    H: tl.constexpr,
    W: tl.constexpr,
    x_stride_b: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    w_stride_oc: tl.constexpr,
    w_stride_ic: tl.constexpr,
    y_stride_b: tl.constexpr,
    y_stride_c: tl.constexpr,
    y_stride_h: tl.constexpr,
    y_stride_w: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)     # batch
    pid_oc = tl.program_id(1)    # output channel
    pid_hw = tl.program_id(2)    # linear index over H*W blocks

    # decode h, w-block
    num_w_blocks = (W + BLOCK_W - 1) // BLOCK_W
    h = pid_hw // num_w_blocks
    wb = pid_hw % num_w_blocks
    w0 = wb * BLOCK_W

    offs_w = w0 + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # loop over input channels
    for ic in range(0, Cin):
        # x[b, ic, h, w]
        x_ix = x_ptr + pid_b * x_stride_b + ic * x_stride_c + h * x_stride_h + offs_w * x_stride_w
        x_val = tl.load(x_ix, mask=mask_w, other=0.0)

        # w[oc, ic]
        w_ix = w_ptr + pid_oc * w_stride_oc + ic * w_stride_ic
        w_val = tl.load(w_ix)  # scalar

        acc += x_val * w_val

    # ReLU6
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    # store y[b, oc, h, w]
    y_ix = y_ptr + pid_b * y_stride_b + pid_oc * y_stride_c + h * y_stride_h + offs_w * y_stride_w
    tl.store(y_ix, acc, mask=mask_w)


# Kernel: Depthwise Conv2d KxK + ReLU6
# Computes: y[b, c, h, w] = relu6( sum_{kh,kw} x[b, c, h+kh, w+kw] * w[c, kh, kw] )
@triton.jit
def depthwise_conv_relu6_kernel(
    x_ptr,         # *f32, [B, C, H, W]
    w_ptr,         # *f32, [C, K, K]
    y_ptr,         # *f32, [B, C, Ho, Wo]
    B: tl.constexpr,
    C: tl.constexpr,      # Hidden
    H: tl.constexpr,
    W: tl.constexpr,
    Ho: tl.constexpr,
    Wo: tl.constexpr,
    K: tl.constexpr,      # kernel size (assume odd)
    pad: tl.constexpr,    # (K - 1) // 2
    x_stride_b: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    w_stride_c: tl.constexpr,
    w_stride_kh: tl.constexpr,
    w_stride_kw: tl.constexpr,
    y_stride_b: tl.constexpr,
    y_stride_c: tl.constexpr,
    y_stride_h: tl.constexpr,
    y_stride_w: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)     # batch
    pid_c = tl.program_id(1)     # channel
    pid_hw = tl.program_id(2)    # over Ho*Wo blocks

    num_w_blocks = (Wo + BLOCK_W - 1) // BLOCK_W
    h = pid_hw // num_w_blocks
    wb = pid_hw % num_w_blocks
    w0 = wb * BLOCK_W

    offs_w = w0 + tl.arange(0, BLOCK_W)
    mask_w = offs_w < Wo

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # conv over kh, kw
    for kh in range(0, K):
        in_h = h + kh - pad
        for kw in range(0, K):
            in_w = offs_w + kw - pad
            valid_w = (in_w >= 0) & (in_w < Wo)
            valid = mask_w & valid_w

            # x[b, c, in_h, in_w]
            x_ix = x_ptr + pid_b * x_stride_b + pid_c * x_stride_c + in_h * x_stride_h + in_w * x_stride_w
            x_val = tl.load(x_ix, mask=valid, other=0.0)

            # w[c, kh, kw]
            w_ix = w_ptr + pid_c * w_stride_c + kh * w_stride_kh + kw * w_stride_kw
            w_val = tl.load(w_ix)  # scalar

            acc += x_val * w_val

    # ReLU6
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    # store
    y_ix = y_ptr + pid_b * y_stride_b + pid_c * y_stride_c + h * y_stride_h + offs_w * y_stride_w
    tl.store(y_ix, acc, mask=mask_w)


# Kernel: 1x1 Conv2d (GEMM) without activation
# Computes: y[b, oc, h, w] = sum_ic x[b, ic, h, w] * w[oc, ic]
@triton.jit
def expand_1x1_kernel(
    x_ptr,         # *f32, [B, Cin, H, W]
    w_ptr,         # *f32, [Cout, Cin]
    y_ptr,         # *f32, [B, Cout, H, W]
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    x_stride_b: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    w_stride_oc: tl.constexpr,
    w_stride_ic: tl.constexpr,
    y_stride_b: tl.constexpr,
    y_stride_c: tl.constexpr,
    y_stride_h: tl.constexpr,
    y_stride_w: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)     # batch
    pid_oc = tl.program_id(1)    # output channel
    pid_hw = tl.program_id(2)    # over H*W blocks

    num_w_blocks = (W + BLOCK_W - 1) // BLOCK_W
    h = pid_hw // num_w_blocks
    wb = pid_hw % num_w_blocks
    w0 = wb * BLOCK_W

    offs_w = w0 + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for ic in range(0, Cin):
        x_ix = x_ptr + pid_b * x_stride_b + ic * x_stride_c + h * x_stride_h + offs_w * x_stride_w
        x_val = tl.load(x_ix, mask=mask_w, other=0.0)
        w_ix = w_ptr + pid_oc * w_stride_oc + ic * w_stride_ic
        w_val = tl.load(w_ix)
        acc += x_val * w_val

    y_ix = y_ptr + pid_b * y_stride_b + pid_oc * y_stride_c + h * y_stride_h + offs_w * y_stride_w
    tl.store(y_ix, acc, mask=mask_w)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.hidden_dim = hidden_dim

        # Parameters shaped like 1x1 Conv weights
        self.expand_weight = nn.Parameter(torch.empty(hidden_dim, in_channels))
        self.dwise_weight = nn.Parameter(torch.empty(hidden_dim, kernel_size, kernel_size))
        self.project_weight = nn.Parameter(torch.empty(out_channels, hidden_dim))

        # Simple uniform init
        bound = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.expand_weight, -bound, bound)
        bound = 1 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.dwise_weight, -bound, bound)
        nn.init.uniform_(self.project_weight, -bound, bound)

    def forward(self, x: torch.Tensor):
        """
        x: (B, Cin, H, W)
        Returns: (B, Cout, Ho, Wo)
        """
        # Fallback if no CUDA / no Triton
        if (not _TRITON_AVAILABLE) or (not x.is_cuda):
            import torch.nn.functional as F
            # expand 1x1 -> Hidden
            w_expand = self.expand_weight.view(1, self.hidden_dim, self.in_channels, 1, 1)
            y = F.conv2d(x, w_expand, bias=None, stride=1, padding=0, groups=1)
            y = F.relu6(y)
            # depthwise KxK
            w_dwise = self.dwise_weight.view(self.hidden_dim, self.kernel_size, self.kernel_size)
            y = F.conv2d(y, w_dwise, bias=None, stride=self.stride, padding=(self.kernel_size - 1)//2, groups=self.hidden_dim)
            y = F.relu6(y)
            # project 1x1
            w_proj = self.project_weight.view(1, self.out_channels, self.hidden_dim, 1, 1)
            out = F.conv2d(y, w_proj, bias=None, stride=1, padding=0, groups=1)
            if self.use_residual:
                out = out + x
            return out

        # CUDA + Triton path
        assert x.dtype == torch.float32, "This Triton implementation assumes float32."
        device = x.device
        x = x.contiguous()

        B, Cin, H, W = x.shape
        Hidden = self.hidden_dim
        K = self.kernel_size
        pad = (K - 1) // 2
        strideconv = self.stride

        # Output sizes after depthwise
        Ho = (H + 2 * pad - K) // strideconv + 1
        Wo = (W + 2 * pad - K) // strideconv + 1

        BLOCK_W = 128
        num_warps = 4

        # 1) expand 1x1 + ReLU6 -> y1: [B, Hidden, H, W]
        y1 = torch.empty((B, Hidden, H, W), device=device, dtype=torch.float32)

        x_strb, x_strc, x_strh, x_strw = x.stride()
        w_expand = self.expand_weight.contiguous()
        w_str_oc, w_str_ic = w_expand.stride()
        y1_strb, y1_strc, y1_strh, y1_strw = y1.stride()

        grid_expand = (B, Hidden, H * _ceil_div(W, BLOCK_W))
        expand_1x1_relu6_kernel[grid_expand](
            x, w_expand, y1,
            B, Cin, Hidden, H, W,
            x_strb, x_strc, x_strh, x_strw,
            w_str_oc, w_str_ic,
            y1_strb, y1_strc, y1_strh, y1_strw,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
        )

        # 2) depthwise KxK + ReLU6 -> y2: [B, Hidden, Ho, Wo]
        y2 = torch.empty((B, Hidden, Ho, Wo), device=device, dtype=torch.float32)

        w_dwise = self.dwise_weight.contiguous().view(Hidden, K, K)  # [C, K, K]
        w_str_c, w_str_kh, w_str_kw = w_dwise.stride()
        y2_strb, y2_strc, y2_strh, y2_strw = y2.stride()

        grid_dwise = (B, Hidden, Ho * _ceil_div(Wo, BLOCK_W))
        depthwise_conv_relu6_kernel[grid_dwise](
            y1, w_dwise, y2,
            B, Hidden, H, W, Ho, Wo,
            K, pad,
            y1_strb, y1_strc, y1_strh, y1_strw,
            w_str_c, w_str_kh, w_str_kw,
            y2_strb, y2_strc, y2_strh, y2_strw,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
        )

        # 3) project 1x1 (no activation) -> out: [B, Cout, Ho, Wo]
        out = torch.empty((B, self.out_channels, Ho, Wo), device=device, dtype=torch.float32)

        w_proj = self.project_weight.contiguous()  # [Cout, Hidden]
        w_str_oc, w_str_ic = w_proj.stride()
        out_strb, out_strc, out_strh, out_strw = out.stride()

        grid_proj = (B, self.out_channels, Ho * _ceil_div(Wo, BLOCK_W))
        expand_1x1_kernel[grid_proj](
            y2, w_proj, out,
            B, Hidden, self.out_channels, Ho, Wo,
            y2_strb, y2_strc, y2_strh, y2_strw,
            w_str_oc, w_str_ic,
            out_strb, out_strc, out_strh, out_strw,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
        )

        # 4) Residual (post proj) if enabled
        if self.use_residual:
            out = out + x

        return out
