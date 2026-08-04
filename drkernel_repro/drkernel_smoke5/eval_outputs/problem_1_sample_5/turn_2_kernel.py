import math
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def conv2d_nchw_stride1_no_pad_dilation1_groups1_kernel(
    x_ptr, w_ptr, y_ptr,
    # shapes
    N: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    # strides (in elements)
    x_sN: tl.constexpr, x_sC: tl.constexpr, x_sH: tl.constexpr, x_sW: tl.constexpr,
    w_sCo: tl.constexpr, w_sCi: tl.constexpr, w_sKH: tl.constexpr, w_sKW: tl.constexpr,
    y_sN: tl.constexpr, y_sC: tl.constexpr, y_sH: tl.constexpr, y_sW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid0 = tl.program_id(0)  # over N*C_out
    pid1 = tl.program_id(1)  # over OW tiles

    # decode n, co
    co = pid0 % C_out
    n  = pid0 // C_out

    # tile start
    start_w = pid1 * BLOCK_W
    offs_w  = start_w + tl.arange(0, BLOCK_W)
    mask_w  = offs_w < OW

    # base pointers for this (n, co) row
    y_row_base = y_ptr + n * y_sN + co * y_sC
    # accumulator in fp32
    outv = tl.zeros([BLOCK_W], dtype=tl.float32)

    # loop over input channels and kernel height/width
    # Note: stride=1, padding=0, dilation=1
    for ci in range(0, C_in):
        for ky in range(0, KH):
            # oh in [0, OH); ih = oh + ky in [ky, ky+OH)
            # x_h = oh + ky
            x_h = ky
            for kx in range(0, KW):
                # load weight scalar w[co, ci, ky, kx]
                w_idx = co * w_sCo + ci * w_sCi + ky * w_sKH + kx * w_sKW
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)

                # x indices: ih = x_h, iw = offs_w + kx
                x_ih = x_h
                x_iw = offs_w + kx
                # compute linear x indices: n*sN + ci*sC + ih*sH + iw*sW
                x_idx = n * x_sN + ci * x_sC + x_ih * x_sH + x_iw * x_sW
                x_val = tl.load(x_ptr + x_idx, mask=mask_w, other=0.0).to(tl.float32)

                # FMA
                outv += w_val * x_val

    # store result
    y_idx = y_row_base + offs_w * y_sW
    tl.store(y_ptr + y_idx, outv, mask=mask_w)


class _Conv2dStride1NoPadDilation1Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        """
        x: (N, C_in, H, W), float32, contiguous
        weight: (C_out, C_in, KH, KW), float32, contiguous
        bias: (C_out,) or None
        Returns: y: (N, C_out, H, W)
        Constraints: stride=1, padding=0, dilation=1, groups=1
        """
        assert x.is_cuda, "Triton kernel requires CUDA tensor"
        assert weight.is_cuda, "Weight must be CUDA"
        assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Only float32 supported in this kernel"
        assert x.is_contiguous(), "x must be contiguous"
        assert weight.is_contiguous(), "weight must be contiguous"

        N, C_in, H, W = x.shape
        C_out, C_in_w, KH, KW = weight.shape
        assert C_in == C_in_w, f"Channel mismatch: x.C={C_in} w.Cin={C_in_w}"
        assert KH == KW, "This kernel assumes square kernels for simplicity"
        # output shape
        OH = H - KH + 1
        OW = W - KW + 1
        assert OH > 0 and OW > 0, f"Invalid output size from H={H}, W={W}, KH={KH}, KW={KW}"
        y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

        # strides in elements
        x_sN, x_sC, x_sH, x_sW = x.stride(0), x.stride(1), x.stride(2), x.stride(3)
        w_sCo, w_sCi, w_sKH, w_sKW = weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3)
        y_sN, y_sC, y_sH, y_sW = y.stride(0), y.stride(1), y.stride(2), y.stride(3)

        # block and grid
        BLOCK_W = 128
        grid = (N * C_out, triton.cdiv(OW, BLOCK_W))

        conv2d_nchw_stride1_no_pad_dilation1_groups1_kernel[grid](
            x, weight, y,
            N, C_in, H, W,
            C_out, KH, KW,
            OH, OW,
            x_sN, x_sC, x_sH, x_sW,
            w_sCo, w_sCi, w_sKH, w_sKW,
            y_sN, y_sC, y_sH, y_sW,
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )

        if bias is not None:
            # broadcast add: y[n, co, oh, ow] += bias[co]
            # Simple torch add is fine here; could be fused but out of scope.
            y = y + bias.view(1, -1, 1, 1)

        # Save for backward (we’ll use PyTorch to compute grads)
        ctx.save_for_backward(x, weight)
        ctx.bias = bias
        ctx.shape_ctx = (N, C_in, H, W, C_out, KH, KW, OH, OW)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        N, C_in, H, W, C_out, KH, KW, OH, OW = ctx.shape_ctx
        bias = ctx.bias

        # Use PyTorch to compute gradients (correct and simpler)
        # Recompute output with autograd enabled to get grads
        x_ = x.detach().requires_grad_(True)
        w_ = weight.detach().requires_grad_(True)
        b_ = None
        if bias is not None:
            b_ = bias.detach().requires_grad_(True)

        # Build a dummy conv to use autograd
        conv = torch.nn.functional.conv2d(x_, w_, bias=b_, stride=1, padding=0, dilation=1, groups=1)
        # Conv output shape is (N, C_out, H - KH + 1, W - KW + 1)
        # This matches our forward
        conv.backward(grad_y)

        grad_x = x_.grad
        grad_w = w_.grad
        grad_b = b_.grad if b_ is not None else None

        return grad_x, grad_w, grad_b


class Conv2dStride1NoPadDilation1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        assert isinstance(kernel_size, int), "Only square kernels supported in this example"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Initialize like torch.nn.Conv2d default
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        # Kaiming uniform init (close to default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if not x.is_cuda or not _HAS_TRITON:
            # Fallback to PyTorch
            return torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=1, padding=0, dilation=1, groups=1)
        return _Conv2dStride1NoPadDilation1Func.apply(x, self.weight, self.bias)
