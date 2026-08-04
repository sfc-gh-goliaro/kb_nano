import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def softmax_tanh_scale_kernel(
    x_ptr,                      # *const float, input after mean + bias: shape [B, C, 1, H, W]
    out_ptr,                    # *float, output: same shape
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,     # not used (D=1), but keep for API symmetry
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    scale: tl.constexpr,        # scaling factor
    BLOCK_C: tl.constexpr
):
    # Program ids: we launch a 2D grid
    # pid0 over B*H*W, pid1 over channel blocks (usually 1)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Decode (b, h, w) from pid0
    HW = H * W
    b = pid0 // HW
    rem = pid0 % HW
    h = rem // W
    w = rem % W

    # Channel block start
    c0 = pid1 * BLOCK_C
    c = c0 + tl.arange(0, BLOCK_C)
    mask = c < C

    # Base pointer for this (b, h, w)
    # Note: D dimension is 1 after mean, so no separate d offset
    base = b * stride_b + h * stride_h + w * stride_w

    # Compute offsets for all channels in the block
    offs = base + c * stride_c

    # Load values; use -inf for masked lanes so they don't affect max
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))

    # Compute in float32 for stability
    x = x.to(tl.float32)

    # Numerically stable softmax: subtract max
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    soft = num / denom

    # Apply tanh and scaling
    # tanh is accurate in [0,1]; do in float32
    tanh_soft = tl.tanh(soft)
    out_val = tanh_soft * scale

    # Store back (cast to input dtype if needed); here assume float32
    tl.store(out_ptr + offs, out_val, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model:
    - Keeps ConvTranspose3d (cuDNN).
    - Fuses softmax + tanh + scaling into a single Triton kernel over (B, C, H, W).
    - Mean over depth and bias add remain as torch ops (simple and fast).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Keep bias as in original: shape [1, C, 1, 1, 1]
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # 1) Transposed convolution (cuDNN)
        y = self.conv_transpose(x)  # [B, C, D, H, W]

        # 2) Mean over depth -> [B, C, 1, H, W]
        # Keep it simple and numerically sound
        y = y.mean(dim=2, keepdim=True)

        # 3) Add bias (broadcast)
        y = y + self.bias

        # If not CUDA or Triton not available, fall back to torch ops
        if (not y.is_cuda) or (not TRITON_AVAILABLE):
            # softmax over channels, then tanh, then scale
            y = torch.softmax(y, dim=1)
            y = torch.tanh(y)
            y = y * self.scaling_factor
            return y

        # Ensure contiguous for predictable strides
        y = y.contiguous()

        # Extract shapes and strides
        B, C, D, H, W = y.shape
        # D should be 1 here
        assert D == 1, f"Expected D=1 after mean-pool over depth, got D={D}"

        # Get strides in elements
        stride_b, stride_c, stride_d, stride_h, stride_w = y.stride()

        # Allocate output
        out = torch.empty_like(y, dtype=torch.float32)  # compute in fp32

        # Choose BLOCK_C as next power-of-two >= C, capped
        block_c = 1 << (int(C - 1).bit_length())
        block_c = min(block_c, 1024)

        # Grid: (B*H*W, ceil_div(C, BLOCK_C))
        grid = (B * H * W, (C + block_c - 1) // block_c)

        # Launch kernel
        softmax_tanh_scale_kernel[grid](
            y, out,
            B, C, H, W,
            stride_b, stride_c, stride_d, stride_h, stride_w,
            self.scaling_factor,
            BLOCK_C=block_c,
            num_warps=4,  # reasonable default; can tune
            num_stages=2
        )

        return out
