import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _scale_add_bias_kernel(
    Y,            # *ptr* to [M, N] output (can be the same as input)
    BIAS,         # *ptr* to [N] bias
    M, N,         # dimensions
    stride_ym,    # row stride of Y
    stride_yn,    # col stride of Y
    scale,        # float: s+1
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Create 2D indices
    mm = offs_m[:, None]
    nn = offs_n[None, :]

    # Pointer to tile
    y_ptrs = Y + mm * stride_ym + nn * stride_yn
    # Combined mask
    mask = (mm < M) & (nn < N)

    # Load y tile
    y = tl.load(y_ptrs, mask=mask, other=0.0)

    # Load bias as vector over N, then broadcast across rows
    bias = tl.load(BIAS + offs_n, mask=mask_n, other=0.0)
    bias = bias[None, :]  # shape (1, BLOCK_N) for broadcast

    # Fused compute: y = scale * y + bias
    out = y * scale + bias

    # Store back
    tl.store(y_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model:
    - Keeps cuBLAS matmul via nn.Linear
    - Fuses scaling and bias addition into a single Triton elementwise kernel
      computing y = (scaling_factor + 1) * y + b
    Entry point class name: ModelNew
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matmul using cuBLAS
        y = self.linear(x)  # shape [B, O]

        # If not CUDA, fallback to torch ops
        if not y.is_cuda:
            scale = self.scaling_factor + 1.0
            return y * scale + self.linear.bias

        # Ensure contiguous for predictable strides
        if not y.is_contiguous():
            y = y.contiguous()

        B, O = y.shape
        bias = self.linear.bias
        if not bias.is_contiguous():
            bias = bias.contiguous()

        # Strides in elements
        stride_ym = y.stride(0)
        stride_yn = y.stride(1)

        # Choose block sizes
        BLOCK_M = 128
        BLOCK_N = 128

        grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(O, BLOCK_N))

        scale = self.scaling_factor + 1.0

        _scale_add_bias_kernel[grid](
            y,                  # in/out
            bias,               # bias
            B, O,
            stride_ym, stride_yn,
            scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

        return y
