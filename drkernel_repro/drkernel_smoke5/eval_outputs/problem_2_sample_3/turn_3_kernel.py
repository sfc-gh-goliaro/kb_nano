import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # tanh(x) = sign(x) * (1 - t) / (1 + t), t = exp(-2*|x|)
    ax = tl.abs(x)
    t = tl.exp(-2.0 * ax)
    num = 1.0 - t
    den = 1.0 + t
    tanh_abs = num / den
    sign = tl.where(x >= 0, 1.0, -1.0)
    y = sign * tanh_abs

    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Stable sigmoid:
    # if x >= 0: s = 1 / (1 + exp(-x))
    # else:      s = exp(x) / (1 + exp(x))
    is_pos = x >= 0
    z = tl.where(is_pos, -x, x)
    e = tl.exp(z)
    s_pos = 1.0 / (1.0 + e)          # when x >= 0, e = exp(-x)
    s_neg = e / (1.0 + e)             # when x < 0,  e = exp(x)
    y = tl.where(is_pos, s_pos, s_neg)

    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Drop-in replacement for the original Model:
    - Keeps attributes: dynamic_thresholded, tanh_params, sigmoid_params
    - forward(x):
        if dynamic_thresholded: return tanh(x) elementwise
        else: return sigmoid(x) elementwise
    - Uses Triton kernels on CUDA; falls back to torch ops otherwise.
    """
    def __init__(self, dynamic_thresholded: bool, tanh_params=None, sigmoid_params=None):
        super().__init__()
        self.dynamic_thresholded = bool(dynamic_thresholded)
        # Keep params to mirror state (not used in computation but for parity)
        self.tanh_params = tanh_params
        self.sigmoid_params = sigmoid_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to torch if CPU or Triton not available
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            if self.dynamic_thresholded:
                return torch.tanh(x)
            else:
                return torch.sigmoid(x)

        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        y = torch.empty_like(x)
        n = x.numel()

        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)

        if self.dynamic_thresholded:
            tanh_kernel[grid](
                x, y,
                n,
                BLOCK_SIZE=BLOCK,
                num_warps=4,
                num_stages=2,
            )
        else:
            sigmoid_kernel[grid](
                x, y,
                n,
                BLOCK_SIZE=BLOCK,
                num_warps=4,
                num_stages=2,
            )

        return y
