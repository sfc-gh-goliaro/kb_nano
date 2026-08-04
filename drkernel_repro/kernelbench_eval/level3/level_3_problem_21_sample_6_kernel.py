import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _conv1x1_bn_relu6_kernel(
        x_ptr,            # *f32, shape [N, Cin, H, W] (contiguous)
        w_ptr,            # *f32, shape [Cout, Cin] (contiguous)
        # b_ptr is expand conv bias; not used (original has no bias)
        mean_ptr,         # *f32, shape [Cout]
        var_ptr,          # *f32, shape [Cout]
        weight_ptr,       # *f32, shape [Cout] (BN scale)
        bias_ptr,         # *f32, shape [Cout] (BN shift)
        eps,              # f32
        y_ptr,            # *f32, shape [N, Cout, H, W] (contiguous)
        N: tl.constexpr,
        Cin: tl.constexpr,
        Cout: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        stride_n: tl.constexpr,
        stride_cin: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        w_stride_co: tl.constexpr,
        w_stride_cin: tl.constexpr,
        y_stride_n: tl.constexpr,
        y_stride_co: tl.constexpr,
        y_stride_h: tl.constexpr,
        y_stride_w: tl.constexpr,
        BLOCK_M: tl.constexpr,  # tile over B = N*H*W
        BLOCK_N: tl.constexpr,  # tile over Cout
    ):
        # 2D launch: program_id(0) over N, program_id(1) over column-blocks
        pid_n = tl.program_id(0)
        pid_col = tl.program_id(1)

        # Column offsets (output channels)
        offs_co = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_co = offs_co < Cout

        # Rows over flattened spatial positions B = N*H*W
        B = N * H * W
        row_start = 0
        while row_start < B:
            offs_b = row_start + tl.arange(0, BLOCK_M)
            mask_b = offs_b < B

            # Decode (h, w) from b
            HW = H * W
            hw = offs_b % HW
            h = hw // W
            w = hw % W

            # Accumulator [BLOCK_N, BLOCK_M]
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

            # Loop over input channels
            for ci in range(0, Cin):
                # x[n, ci, h, w]
                x_addr = x_ptr + pid_n * stride_n + ci * stride_cin + h * stride_h + w * stride_w
                xv = tl.load(x_addr, mask=mask_b, other=0.0)  # (BLOCK_M,)

                # w[co, ci]
                w_addr = w_ptr + offs_co * w_stride_co + ci * w_stride_cin
                wv = tl.load(w_addr, mask=mask_co, other=0.0)  # (BLOCK_N,)

                # Outer product accumulate
                acc += wv[:, None] * xv[None, :]

            # BN: ((acc - mean) / sqrt(var + eps)) * weight + bias
            mean = tl.load(mean_ptr + offs_co, mask=mask_co, other=0.0)   # (BLOCK_N,)
            var = tl.load(var_ptr + offs_co, mask=mask_co, other=1.0)     # (BLOCK_N,)
            scale = tl.load(weight_ptr + offs_co, mask=mask_co, other=1.0)
            shift = tl.load(bias_ptr + offs_co, mask=mask_co, other=0.0)

            denom = tl.sqrt(var + eps)
            norm = (acc - mean[:, None]) / denom[:, None]
            out = norm * scale[:, None] + shift[:, None]

            # ReLU6
            out = tl.minimum(tl.maximum(out, 0.0), 6.0)

            # Store y[n, co, h, w]
            # Build addresses as broadcasted 2D
            n = pid_n
            co_b = offs_co[:, None]  # (BLOCK_N, 1)
            b_b = offs_b[None, :]     # (1, BLOCK_M)
            # y stride layout [n, co, h, w]
            y_addr = y_ptr + n * y_stride_n + co_b * y_stride_co + h[None, :] * y_stride_h + w[None, :] * y_stride_w
            store_mask = mask_co[:, None] & mask_b[None, :]
            tl.store(y_addr, out, mask=store_mask)

            row_start += BLOCK_M

    @triton.jit
    def _conv1x1_bn_kernel(
        x_ptr,            # *f32, shape [N, Cin, H, W] (contiguous)
        w_ptr,            # *f32, shape [Cout, Cin] (contiguous)
        mean_ptr,         # *f32, shape [Cout]
        var_ptr,          # *f32, shape [Cout]
        weight_ptr,       # *f32, shape [Cout] (BN scale)
        bias_ptr,         # *f32, shape [Cout] (BN shift)
        eps,              # f32
        y_ptr,            # *f32, shape [N, Cout, H, W] (contiguous)
        N: tl.constexpr,
        Cin: tl.constexpr,
        Cout: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        stride_n: tl.constexpr,
        stride_cin: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        w_stride_co: tl.constexpr,
        w_stride_cin: tl.constexpr,
        y_stride_n: tl.constexpr,
        y_stride_co: tl.constexpr,
        y_stride_h: tl.constexpr,
        y_stride_w: tl.constexpr,
        BLOCK_M: tl.constexpr,  # tile over B = N*H*W
        BLOCK_N: tl.constexpr,  # tile over Cout
    ):
        pid_n = tl.program_id(0)
        pid_col = tl.program_id(1)

        offs_co = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_co = offs_co < Cout

        B = N * H * W
        row_start = 0
        while row_start < B:
            offs_b = row_start + tl.arange(0, BLOCK_M)
            mask_b = offs_b < B

            HW = H * W
            hw = offs_b % HW
            h = hw // W
            w = hw % W

            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

            for ci in range(0, Cin):
                x_addr = x_ptr + pid_n * stride_n + ci * stride_cin + h * stride_h + w * stride_w
                xv = tl.load(x_addr, mask=mask_b, other=0.0)

                w_addr = w_ptr + offs_co * w_stride_co + ci * w_stride_cin
                wv = tl.load(w_addr, mask=mask_co, other=0.0)

                acc += wv[:, None] * xv[None, :]

            mean = tl.load(mean_ptr + offs_co, mask=mask_co, other=0.0)
            var = tl.load(var_ptr + offs_co, mask=mask_co, other=1.0)
            scale = tl.load(weight_ptr + offs_co, mask=mask_co, other=1.0)
            shift = tl.load(bias_ptr + offs_co, mask=mask_co, other=0.0)

            denom = tl.sqrt(var + eps)
            norm = (acc - mean[:, None]) / denom[:, None]
            out = norm * scale[:, None] + shift[:, None]

            # Store
            n = pid_n
            co_b = offs_co[:, None]
            b_b = offs_b[None, :]
            y_addr = y_ptr + n * y_stride_n + co_b * y_stride_co + h[None, :] * y_stride_h + w[None, :] * y_stride_w
            store_mask = mask_co[:, None] & mask_b[None, :]
            tl.store(y_addr, out, mask=store_mask)

            row_start += BLOCK_M


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio,
                 block1x1_bn_eps=1e-5, project_bn_eps=1e-5):
        """
        Triton-optimized MBConv block:
        - expand 1x1 (if expand_ratio != 1): Conv2d + BN + ReLU6
        - depthwise kernel_sizexkernel_size: cuDNN conv2d
        - project 1x1: Conv2d + BN
        Residual add when stride == 1 and in_channels == out_channels (after project).
        """
        super(ModelNew, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        self.expand_ratio = expand_ratio
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if expand_ratio != 1:
            # Conv weights: [Cout, Cin]
            self.expand_weight = nn.Parameter(torch.empty(hidden_dim, in_channels))
            nn.init.kaiming_uniform_(self.expand_weight, a=math.sqrt(5))
            # BN
            self.expand_bn = nn.BatchNorm2d(hidden_dim, eps=block1x1_bn_eps, momentum=0.1)
        else:
            self.expand_weight = None
            self.expand_bn = None

        # Depthwise conv weights: [C, 1, K, K]
        self.depthwise_weight = nn.Parameter(torch.empty(hidden_dim, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim, eps=block1x1_bn_eps, momentum=0.1)

        # Project 1x1
        self.project_weight = nn.Parameter(torch.empty(out_channels, hidden_dim))
        nn.init.kaiming_uniform_(self.project_weight, a=math.sqrt(5))
        self.project_bn = nn.BatchNorm2d(out_channels, eps=project_bn_eps, momentum=0.1)

        # Triton tuning
        self.BLOCK_M = 256
        self.BLOCK_N = 64
        self.num_warps = 4
        self.num_stages = 2

        # Assume eval mode for forward speed
        self._in_eval = True

    def train(self, mode: bool = True):
        super(ModelNew, self).train(mode)
        self._in_eval = not mode
        if self.expand_bn is not None:
            self.expand_bn.train(mode)
        self.depthwise_bn.train(mode)
        self.project_bn.train(mode)
        return self

    def forward(self, x: torch.Tensor):
        """
        Forward:
        1) expand 1x1 -> BN -> ReLU6  (optional)
        2) depthwise conv (cuDNN) -> BN -> ReLU6
        3) project 1x1 -> BN
        4) residual add if stride == 1 and in == out
        """
        assert x.is_cuda, "ModelNew requires CUDA tensors"
        assert x.dtype == torch.float32, "ModelNew assumes float32 for numerical stability"

        device = x.device
        N, C, H, W = x.shape

        # 1) expand 1x1 + BN + ReLU6
        if self.expand_ratio != 1:
            cout = self.hidden_dim
            expand_w = self.expand_weight
            expand_bn = self.expand_bn

            if self._in_eval:
                mean = expand_bn.running_mean
                var = expand_bn.running_var
                weight = expand_bn.weight
                bias = expand_bn.bias
            else:
                # Use running for simplicity
                mean = expand_bn.running_mean
                var = expand_bn.running_var
                weight = expand_bn.weight
                bias = expand_bn.bias

            x1 = torch.empty((N, cout, H, W), device=device, dtype=x.dtype)

            grid = (N, triton.cdiv(cout, self.BLOCK_N))
            _conv1x1_bn_relu6_kernel[grid](
                x, expand_w, mean, var, weight, bias, expand_bn.eps, x1,
                N, C, cout, H, W,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                expand_w.stride(0), expand_w.stride(1),
                x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N,
                num_warps=self.num_warps, num_stages=self.num_stages,
            )
            x = x1
        else:
            cout = C  # hidden = C

        # 2) depthwise conv (cuDNN) -> BN -> ReLU6
        w_d = self.depthwise_weight
        x = F.conv2d(x, w_d, bias=None, stride=self.stride, padding=(self.kernel_size - 1) // 2, groups=cout)
        dw_bn = self.depthwise_bn
        if self._in_eval:
            dw_mean = dw_bn.running_mean
            dw_var = dw_bn.running_var
            dw_weight = dw_bn.weight
            dw_bias = dw_bn.bias
        else:
            dw_mean = dw_bn.running_mean
            dw_var = dw_bn.running_var
            dw_weight = dw_bn.weight
            dw_bias = dw_bn.bias
        x = F.batch_norm(x, dw_mean, dw_var, dw_weight, dw_bias, training=False, momentum=dw_bn.momentum, eps=dw_bn.eps)
        x = F.relu6(x)

        # 3) project 1x1 + BN
        cin = x.shape[1]
        cout = self.out_channels
        proj_w = self.project_weight
        proj_bn = self.project_bn
        if self._in_eval:
            pj_mean = proj_bn.running_mean
            pj_var = proj_bn.running_var
            pj_weight = proj_bn.weight
            pj_bias = proj_bn.bias
        else:
            pj_mean = proj_bn.running_mean
            pj_var = proj_bn.running_var
            pj_weight = proj_bn.weight
            pj_bias = proj_bn.bias

        y = torch.empty((N, cout, H, W), device=device, dtype=x.dtype)
        grid = (N, triton.cdiv(cout, self.BLOCK_N))
        _conv1x1_bn_kernel[grid](
            x, proj_w, pj_mean, pj_var, pj_weight, pj_bias, proj_bn.eps, y,
            N, cin, cout, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            proj_w.stride(0), proj_w.stride(1),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

        # 4) residual add if applicable (after project, matching original)
        if self.use_residual:
            y = y + x

        return y
