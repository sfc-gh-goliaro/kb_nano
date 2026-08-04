import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def fused_mean_softmax_tanh_scale_kernel(
    x_ptr,             # *f32, shape [B, C, D, H, W]
    bias_ptr,          # *f32, shape [C] (we'll pass a view)
    out_ptr,           # *f32, shape [B, C, 1, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    sN: tl.constexpr,  # stride for N in x
    sC: tl.constexpr,  # stride for C in x
    sD: tl.constexpr,  # stride for D in x
    sH: tl.constexpr,  # stride for H in x
    sW: tl.constexpr,  # stride for W in x
    osN: tl.constexpr, # stride for N in out
    osC: tl.constexpr, # stride for C in out
    osH: tl.constexpr, # stride for H in out
    osW: tl.constexpr, # stride for W in out
    scale: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program ids
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Bounds check
    if (pid_b >= B) or (pid_h >= H) or (pid_w >= W):
        return

    # --- Step 1: compute mean over D for each channel c ---
    # z[c] = (1/D) * sum_d x[b, c, d, h, w]
    z = tl.zeros([C], dtype=tl.float32)
    d = 0
    while d < D:
        # x index: b*sN + c*sC + d*sD + h*sH + w*sW
        ptr = x_ptr + pid_b * sN + d * sD + pid_h * sH + pid_w * sW + tl.arange(0, C) * sC
        vals = tl.load(ptr)  # shape [C]
        z += vals
        d += 1
    z = z / D

    # --- Step 2: add bias ---
    # bias shape [C]; out = z + bias
    b_ptr = bias_ptr + tl.arange(0, C)
    bias_vals = tl.load(b_ptr)
    z = z + bias_vals

    # --- Step 3: softmax over channels (numerically stable) ---
    # max over c
    m = -float('inf')
    c = 0
    while c < C:
        chunk = z[c:c + BLOCK_C]
        chunk_max = tl.max(chunk, axis=0)
        m = tl.maximum(m, chunk_max)
        c += BLOCK_C

    # sum exp(z - m)
    sum_e = 0.0
    c = 0
    while c < C:
        chunk = z[c:c + BLOCK_C]
        sum_e += tl.sum(tl.exp(chunk - m), axis=0)
        c += BLOCK_C

    inv_sum = 1.0 / sum_e

    # --- Step 4: write output y = tanh(scale * softmax), softmax = exp(z - m) / sum_e ---
    c = 0
    while c < C:
        chunk = z[c:c + BLOCK_C]
        e = tl.exp(chunk - m)
        soft = e * inv_sum
        out_vals = tl.tanh(soft * scale)
        # Store to out[b, c, 0, h, w]
        out_ptr_c = out_ptr + pid_b * osN + pid_h * osH + pid_w * osW + tl.arange(0, BLOCK_C) * osC + c
        tl.store(out_ptr_c, out_vals, mask=tl.arange(0, BLOCK_C) + c < C)
        c += BLOCK_C


class ModelNew(nn.Module):
    """
    Triton-optimized version:
      - Keep ConvTranspose3d in PyTorch (cuDNN)
      - Fuse mean-over-depth + bias-add + softmax-over-channels + tanh + scale into a single Triton kernel
    Entry point name: ModelNew
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # bias is shape (1, C, 1, 1, 1); we will use a view (C,) in the kernel
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor):
        # 1) ConvTranspose3d (cuDNN)
        x = self.conv_transpose(x)  # shape (B, C, D, H, W)
        if not x.is_cuda:
            # Fallback to pure PyTorch if not on GPU
            x = x.mean(dim=2, keepdim=True)
            x = x + self.bias
            x = torch.softmax(x, dim=1)
            x = torch.tanh(x)
            x = x * self.scaling_factor
            return x

        # Ensure dtype float32 for numerical stability
        if x.dtype != torch.float32:
            x = x.float()

        B, C, D, H, W = x.shape
        device = x.device

        # 2) Mean over depth using our fused kernel
        # Prepare output
        out = torch.empty((B, C, 1, H, W), dtype=torch.float32, device=device)

        # Get strides in elements
        sN, sC, sD, sH, sW = x.stride()
        osN, osC, _, osH, osW = out.stride()

        # Bias as (C,) view
        bias_1d = self.bias.view(-1)
        if bias_1d.dtype != torch.float32:
            bias_1d = bias_1d.float()

        # Choose block size for channels
        BLOCK_C = 128 if C >= 128 else (64 if C >= 64 else 32)

        grid = (B, H, W)
        fused_mean_softmax_tanh_scale_kernel[grid](
            x, bias_1d, out,
            B, C, D, H, W,
            sN, sC, sD, sH, sW,
            osN, osC, osH, osW,
            self.scaling_factor,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        return out
