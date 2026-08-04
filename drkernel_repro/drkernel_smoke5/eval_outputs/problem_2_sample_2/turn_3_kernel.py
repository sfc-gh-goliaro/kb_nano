import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def mean_depth_vec_kernel(
    x_ptr, y_ptr,
    B, C, D, H, W,
    sN, sC, sD, sH, sW,
    out_sN, out_sC, out_sD, out_sH, out_sW,
    BLOCK_W: tl.constexpr,
):
    # 2D grid:
    #  - pid0 indexes over (b, c, h)
    #  - pid1 indexes over width blocks
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    CH = C * H
    b = pid0 // CH
    rem = pid0 % CH
    c = rem // H
    h = rem % H

    w_start = pid1 * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask = offs_w < W

    # base pointer for d=0, w block
    base = b * sN + c * sC + h * sH + offs_w * sW

    # fp32 accumulator for this block
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # loop over depth, accumulate vector
    d = 0
    while d < D:
        ptrs = x_ptr + base + d * sD
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += vals.to(tl.float32)
        d += 1

    mean = acc / D

    # store to y at d=0
    out_ptrs = y_ptr + b * out_sN + c * out_sC + h * out_sH + offs_w * out_sW
    tl.store(out_ptrs, mean, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model:
    - Keep cuDNN ConvTranspose3d
    - Replace mean over depth with a fast, vectorized Triton kernel
    -其余操作保持PyTorch
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # 1) ConvTranspose3d (cuDNN)
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 2) Mean over depth using Triton -> y: (B, C, 1, H, W)
        if not x.is_cuda:
            y = x.mean(dim=2, keepdim=True)
        else:
            # Ensure contiguous for best performance (sW=1)
            if not x.is_contiguous():
                x = x.contiguous()

            y = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)

            sN, sC, sD, sH, sW = x.stride()
            out_sN, out_sC, out_sD, out_sH, out_sW = y.stride()

            # Choose block size and launch params
            BLOCK_W = 128 if W >= 128 else 64
            grid = (B * C * H, triton.cdiv(W, BLOCK_W))

            mean_depth_vec_kernel[grid](
                x, y,
                B, C, D, H, W,
                sN, sC, sD, sH, sW,
                out_sN, out_sC, out_sD, out_sH, out_sW,
                BLOCK_W=BLOCK_W,
                num_warps=4,
                num_stages=2,
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
