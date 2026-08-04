import torch
import triton
import triton.language as tl


# Elementwise add kernel: out[i] = x[i] + y[i]
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized elementwise add: y = x + y
    - CUDA tensors: uses a fused Triton kernel.
    - CPU or non-CUDA: falls back to torch.add.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Validate shapes
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")

        # CPU / non-CUDA fallback
        if not x.is_cuda or not y.is_cuda:
            return x + y

        # Dtype handling: compute in float32 for stability; cast back
        in_dtype = x.dtype
        if not x.dtype.is_floating_point or not y.dtype.is_floating_point:
            x32 = x.to(torch.float32)
            y32 = y.to(torch.float32)
        else:
            x32 = x
            y32 = y

        # Contiguity
        x32 = x32.contiguous()
        y32 = y32.contiguous()

        n = x32.numel()
        out32 = torch.empty_like(x32)

        # Launch config
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)

        add_kernel[grid](
            x32, y32, out32,
            n_elements=n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        # Cast back to original dtype if we upcast
        if out32.dtype != in_dtype:
            return out32.to(in_dtype)
        return out32
