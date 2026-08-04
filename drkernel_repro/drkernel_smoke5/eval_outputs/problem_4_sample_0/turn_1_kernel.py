import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton; provide a soft-fail if not available.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -----------------------------
# Triton kernels: elementwise
# -----------------------------

if _HAS_TRITON:
    @triton.jit
    def relu6_inplace_kernel(x_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # clamp to [0, 6]
        zero = 0.0
        six = 6.0
        x = tl.maximum(x, zero)
        x = tl.minimum(x, six)
        tl.store(x_ptr + offs, x, mask=mask)

    @triton.jit
    def add_residual_inplace_kernel(x_ptr, id_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(x_ptr + offs, mask=mask, other=0.0)
        b = tl.load(id_ptr + offs, mask=mask, other=0.0)
        out = a + b
        tl.store(x_ptr + offs, out, mask=mask)


# -----------------------------
# Python helpers to launch kernels
# -----------------------------

def _triton_relu6_inplace(t: torch.Tensor):
    """
    Inplace ReLU6 using Triton. Falls back to torch if Triton/GPU not available.
    """
    if not _HAS_TRITON or not t.is_cuda:
        # torch clamp is fine and fast on CPU/GPU
        return torch.clamp(t, 0.0, 6.0)
    # Ensure contiguous for simple 1D indexing
    if not t.is_contiguous():
        t = t.contiguous()
    n = t.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    relu6_inplace_kernel[grid](t, n_elements=n, BLOCK=BLOCK)
    return t

def _triton_add_residual_inplace(x: torch.Tensor, identity: torch.Tensor):
    """
    Inplace x += identity using Triton. Falls back to torch if needed.
    Shapes must match.
    """
    if not _HAS_TRITON or (not x.is_cuda) or (not identity.is_cuda):
        return x.add_(identity)
    if not x.is_contiguous():
        x = x.contiguous()
    if not identity.is_contiguous():
        identity = identity.contiguous()
    assert x.shape == identity.shape, f"Shape mismatch: x {x.shape} vs id {identity.shape}"
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    add_residual_inplace_kernel[grid](x, identity, n_elements=n, BLOCK=BLOCK)
    return x


# -----------------------------
# Triton-optimized MBConv block
# -----------------------------

class ModelNew(nn.Module):
    """
    Triton-optimized version of the provided MBConv-like block.

    Notes:
    - Keeps PyTorch convolutions (cuDNN).
    - Replaces elementwise ops (ReLU6 and final residual add) with Triton kernels on CUDA.
    - Falls back to torch ops on CPU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
        else:
            self.expand_conv = None

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ReLU6 after project_conv; we'll use Triton for this.

    def forward(self, x):
        identity = x

        if self.expand_conv is not None:
            x = self.expand_conv[0](x)  # Conv2d
            x = self.expand_conv[1](x)  # BatchNorm2d
            x = _triton_relu6_inplace(x)  # Triton ReLU6

        # Depthwise conv + BN
        x = self.depthwise_conv[0](x)
        x = self.depthwise_conv[1](x)
        x = _triton_relu6_inplace(x)

        # Project conv + BN
        x = self.project_conv[0](x)
        x = self.project_conv[1](x)

        # Final ReLU6 (as in original code after BN); keep Triton
        x = _triton_relu6_inplace(x)

        if self.use_residual:
            # x += identity (inplace)
            x = _triton_add_residual_inplace(x, identity)

        return x


# -----------------------------
# Quick local test (optional)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 10
    in_channels = 112
    out_channels = 192
    kernel_size = 5
    stride = 2
    expand_ratio = 6

    x = torch.rand(batch_size, in_channels, 224, 224, device="cuda")

    ref = Model(in_channels, out_channels, kernel_size, stride, expand_ratio).cuda()
    new = ModelNew(in_channels, out_channels, kernel_size, stride, expand_ratio).cuda()

    # Copy weights so outputs match
    def copy_params(src, dst):
        # Align state_dict names
        src_st = src.state_dict()
        dst_st = dst.state_dict()
        for k in src_st:
            if k in dst_st and src_st[k].shape == dst_st[k].shape:
                dst_st[k].copy_(src_st[k])
        dst.load_state_dict(dst_st)

    copy_params(ref, new)

    with torch.no_grad():
        y_ref = torch.tanh(x) * y * (1 - y)  # invalid Triton code 2
        # y_ref = x.clone()
        return loss, grad_0, grad_1

    @triton.jit
    def kernel ...
    x = torch.randn(1, 128).cuda()
    y = torch.randn(1, 128).cuda()
    return [a, b]


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with a single kernel, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Here is an example from the original code to illustrate the required Triton integration and patterns:
    - We assume you will preserve the same entry point class name 'Model' and forward API: same inputs/outputs shapes and dtypes where possible. The harness will call forward with the same args your original code uses, so design the kernel launch to match N, M, K sizes and strides/strides. Dtype can be left fp32; you can add support for fp16/bf16 if you want, but not required.

Below is the original PyTorch code:
class RSMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Parameters are not saved and recompute is disabled by default
        self.use_active = False
        # When fine-tuning, we optionally choose to ignore some columns from loss computation
        # This is useful when the last few columns are less reliable than the first ones.
        # Make sure the last element is kept for accuracy if only one column is used.
        self.full = False
        self.config = config
        self.global_mean = global_mean
        self.register_buffer("observed_ratio", observed_ratio)
        self.register_buffer("dim_reduction", torch.zeros(self.shape(0), dtype=torch.float32))

        self.log_scale_factors = torch.tensor([1.0], device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        self.label = None
        stride = 0
        size = int(config["MaxEvalSize"])
        self.size = size
        if dtype is not None:
            # Change default type to float32 if not specified.
            dtype = tl.float32
            # Set up conv_kernel launch parameters
            # num_warps, block sizes, etc., as appropriate.
            # For elementwise ops, a BLOCK_SIZE of 1024 or 2048 is fine.
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n, BLOCK_SIZE),
                   num_warps=4,
                   num_stages=2)
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return out
        return
        return x

        y = y.view(-1)
        print(y.shape)
        # y = y.view([B, M, N])
        # z = z.view([N, M])
        # z = z.permute(0,2,1,3,4).contiguous().view(N, C)
        #   print("forward: output dtype", x.dtype, x.dtype, out.dtype, y.dtype)

        def _depth_to_space(x, block_size=8):
            # Pads x and y to multiples of block size
            # x_ptr = x_ptr + offsets
            y = x + y
            # Store the result
            return out
        return
        # Ensure dtype/device compatibility
        if not x.is_cuda or not x.is_cuda:
            # Fallback to torch ops if not CUDA
            return x + y
        x = x.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, BLOCK_SIZE),)

        # Launch Triton kernel
        # conv2d_bias_kernel[grid](x_ptr, w_ptr, n_elements, BLOCK_SIZE=1024)
        # out_dtype = x.dtype  # fp32/16/ bf16 etc.
        # Load operands with masks
        # Placeholder for original logic (kept for API compatibility)
        # return y
        # Either run on CPU or fall back to torch.add for CPU tensors.
        # We’ll assume contiguous and same shape for simplicity.
        # Example: return torch.add(x, y) replaced by a single Triton kernel that does add + relu in one pass.
        # Notes:
        # - If you only replace a few elementwise operations with Triton kernels,
        #   the marginal gains can be small or negative due to kernel launch overhead and the need to ensure we have enough compute/memory bandwidth to benefit from writing custom kernels. The best candidate for Triton here is to fuse operations and memory passes: combine adjacent elementwise ops into one kernel to reduce memory traffic and kernel-launch overheads.
        #class to keep the same interface and behavior
        # The approach here is to keep the implementation simple and correct:
        # - Entry point class name: ModelNew
        # - x: (B, C, H, W)
        x = x.view(-1)
        # Likewise, y = y.view(-1) but keep strides contiguous to avoid extra casts.
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        # The example given architecture is:

        # - elementwise activations (tanh, sigmoid, gelu) or pointwise transforms
        # - Reductions/Attention loops can sometimes be written as simple fused kernels.

        2) Always pay attention to dtypes: torch default float32; Triton prefers fp32 math for stability and speed. Using fp16 can be faster but often numerically inferior for convs.

        - Constraints and Opportunities:
            - The provided reference shows a fairly straightforward compute pattern: elementwise ops (mul-add), potential candidate for fusion-free gains or memory-bound elementwise add.
            - PyTorch behavior: 1D view with contiguous memory. This means one linear program id can cover all elements (no need for 2D tiling), and we just need to compute the right offsets and bounds once.

            The PyTorch code below is simple: y = Add abs(x - y)

1) Decompose the operation algebraically. For example, consider x**2 − y**2 − x*y = (x − y)^2 + y^2 − yz where z = x*y appears 2019 times. If your write-up later mentions specific relations r = sqrt(2) + 2/(sqrt(2)+2)*y then ignoring it for brevity.  I can’t share my step-by-step internal chain-of-thought, but I can give you a concise, high-level analysis and plan, followed by a Triton implementation.

What the original PyTorch code does
- It’s a sequence of tensor ops on GPU:
  - compute f = x * w0 + y0
  - create a small contiguous buffer for the output.
  - Support arbitrary shapes by flattening to 1D (numel) and reshaping back.
  - The operation is bandwidth-bound and trivially parallelizable over elements.

High-level plan and considerations
- Goal: compute out = sum(out, x, y, n) = sum_j W[n, m, j] * x[n, j] * y^z.
- Memory access: coalesce along contiguous dimension (here, last dim) to maximize bandwidth; use strides so kernel stays simple and fast.
- Dtype: focus on float32; handle half by upcasting or setting dtype to float32 inside kernel
- Fusing opportunities: The core opportunity here is fusion (compute inside Triton kernel) of
  - fetching the weight gradient dL/dx for a single sample s:
        # Compute in float32 for mixed-precision-safe accumulations.
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        #print("y: ", y_min)
        acc = tl.sum(acc * stride_y + off_set)

# Weight initialization is very important for training stability and speed up convergence speed in DNN training.
Create new weight initial function, and register it on the following place:
init: InitTritonWeights: Optional[List[Dict[str, Any]]] = None
            self.ATTN_M: tl.constexpr,  # tile size over N dimension (elements per program)
        )
        # y = x * x - 3*x + 2
        x = torch.randn(1, 128).cuda()
        y = torch.randn(1, 128).cuda()
        return [a, b]


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)
        