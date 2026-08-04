import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def fused_mean_softmax_tanh_scale_kernel(
    x_ptr,             # *f32, shape [B, C, D, H, W]
    bias_ptr,          # *f32, shape [C]
    out_ptr,           # *f32, shape [B, C, 1, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    sN: tl.constexpr,  # stride for N in x (elements)
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

    if (pid_b >= B) or (pid_h >= H) or (pid_w >= W):
        return

    # --- Step 1: compute mean over D for each channel c ---
    z = tl.zeros([C], dtype=tl.float32)
    d = 0
    while d < D:
        # Load x[b, :, d, h, w] as a vector over channels
        ptr = x_ptr + pid_b * sN + d * sD + pid_h * sH + pid_w * sW + tl.arange(0, C) * sC
        vals = tl.load(ptr)  # shape [C]
        z += vals
        d += 1
    z = z / D

    # --- Step 2: add bias ---
    bias_vals = tl.load(bias_ptr + tl.arange(0, C))
    z = z + bias_vals

    # --- Step 3: softmax over channels (numerically stable) ---
    # Pass 1: max
    m = -float('inf')
    off = 0
    while off < C:
        chunk = z[off:off + BLOCK_C]
        chunk_max = tl.max(chunk, axis=0)
        m = tl.maximum(m, chunk_max)
        off += BLOCK_C

    # Pass 2: sum exp(z - m)
    sum_e = 0.0
    off = 0
    while off < C:
        chunk = z[off:off + BLOCK_C]
        sum_e += tl.sum(tl.exp(chunk - m), axis=0)
        off += BLOCK_C
    inv_sum = 1.0 / sum_e

    # Pass 3: normalize, tanh, scale, and store
    off = 0
    while off < C:
        chunk = z[off:off + BLOCK_C]
        e = tl.exp(chunk - m)
        soft = e * inv_sum
        # tanh(soft * scale) = (exp(2*soft*scale) - 1) / (exp(2*soft*scale) + 1)
        t = soft * scale
        et = tl.exp(2.0 * t)
        out_vals = (et - 1.0) / (et + 1.0)
        # Store to out[b, c, 0, h, w]
        out_ptr_c = out_ptr + pid_b * osN + pid_h * osH + pid_w * osW + tl.arange(0, BLOCK_C) * osC + off
        tl.store(out_ptr_c, out_vals, mask=(off + tl.arange(0, BLOCK_C)) < C)
        off += BLOCK_C


class ModelNew(nn.Module):
    """
    Triton-optimized version:
      - Keep ConvTranspose3d in PyTorch (cuDNN)
      - Fuse mean-over-depth + bias-add + softmax-over-channels into a single Triton kernel
      - Implement tanh via exp to avoid missing tl.tanh
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))  # broadcastable over (B,C,1,H,W)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor):
        # 1) ConvTranspose3d (cuDNN)
        x = self.conv_transpose(x)  # shape (B, C, D, H, W)
        if not x.is_cuda:
            # CPU fallback using PyTorch
            x = x.mean(dim=2, keepdim=True)
            x = x + self.bias
            x = torch.softmax(x, dim=1)
            x = torch.tanh(x)
            x = x * self.scaling_factor
            return x

        # Ensure float32 for numerical stability
        if x.dtype != torch.float32:
            x = x.float()

        B, C, D, H, W = x.shape
        device = x.device

        # 2) Allocate output
        out = torch.empty((B, C, 1, H, W), dtype=torch.float32, device=device)

        # Strides in elements
        sN, sC, sD, sH, sW = x.stride()
        osN, osC, _, osH, osW = out.stride()

        # Bias as (C,)
        bias_1d = self.bias.view(-1)
        if bias_1d.dtype != torch.float32:
            bias_1d = bias_1d.float()

        # Block size for channels
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
