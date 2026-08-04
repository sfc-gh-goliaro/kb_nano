import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def mean_depth_kernel(
    x_ptr, y_ptr,
    B, C, D, H, W,
    sN, sC, sD, sH, sW,
    out_sN, out_sC, out_sD, out_sH, out_sW,
):
    # one program per output element (b, c, h, w) at d=0
    pid = tl.program_id(0)
    CHW = C * H * W
    b = pid // CHW
    rem = pid % CHW
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    # base input pointer for (b, c, 0, h, w)
    base_in = b * sN + c * sC + h * sH + w * sW
    acc = 0.0  # fp32 accumulation

    # loop over depth
    d = 0
    while d < D:
        val = tl.load(x_ptr + base_in + d * sD)
        acc += val.to(tl.float32)
        d += 1

    mean = acc / D

    # store to y at (b, c, 0, h, w)
    out_ptr = y_ptr + b * out_sN + c * out_sC + h * out_sH + w * out_sW  # d=0
    tl.store(out_ptr, mean)


class ModelNew(nn.Module):
    """
    Triton-optimized version:
    - Keep cuDNN ConvTranspose3d
    - Replace mean over depth with a Triton kernel
    -其余操作保持PyTorch
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        # keep the same bias shape and semantics
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # 1) ConvTranspose3d (cuDNN)
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 2) Mean over depth using Triton -> y: (B, C, 1, H, W)
        if not x.is_cuda:
            # CPU fallback
            y = x.mean(dim=2, keepdim=True)
        else:
            y = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)
            sN, sC, sD, sH, sW = x.stride()
            out_sN, out_sC, out_sD, out_sH, out_sW = y.stride()

            grid = (B * C * H * W,)
            mean_depth_kernel[grid](
                x, y,
                B, C, D, H, W,
                sN, sC, sD, sH, sW,
                out_sN, out_sC, out_sD, out_sH, out_sW,
                num_warps=1,
                num_stages=1,
            )

        # 3) Add bias (broadcast)
        y = y + self.bias

        # 4) Softmax over channels
        y = torch.softmax(y, dim=1)

        # 5) Tanh
        y = torch.tanh(y)

        # 6) Scale
        y = y * self.scaling_factor
        return y
