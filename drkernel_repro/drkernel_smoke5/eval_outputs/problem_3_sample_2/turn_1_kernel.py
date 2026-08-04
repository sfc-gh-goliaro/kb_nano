import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def matmul_bias_scale_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale,  # python float, will be treated as scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for blocks
    A_block_ptr = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_block_ptr = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        if k_remaining > 0:
            k_mask = offs_k < k_remaining
        else:
            k_mask = offs_k < 1  # false

        a = tl.load(A_block_ptr, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_block_ptr, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        # Accumulate
        acc += tl.dot(a, b)

        # Advance pointers
        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk

    # Epilogue: scale and add bias
    # scale is a python float -> Triton scalar; acc is float32
    c = acc * scale
    if Bias is not None:
        # Bias shape is [N]
        bias_vals = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
        # Broadcast over rows
        c += bias_vals[None, :]

    # Store result
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _triton_linear(x: torch.Tensor,
                   weight: torch.Tensor,
                   bias: torch.Tensor,
                   scaling_factor: float) -> torch.Tensor:
    """
    Compute y = (x @ weight.T) * (1 + scaling_factor) + bias using a Triton kernel.

    Args:
        x: [M, K]
        weight: [N, K] in PyTorch Linear storage (so weight.T is [K, N])
        bias: [N] or None
        scaling_factor: float
    Returns:
        y: [M, N]
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert weight.is_cuda, "Weight must be CUDA"
    if bias is not None:
        assert bias.is_cuda, "Bias must be CUDA"

    # Shapes
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, f"weight shape mismatch: got {weight.shape}, expected (*, {K})"

    # Make sure memory is contiguous or pass correct strides
    A = x
    # B should be [K, N]; weight.T is [K, N] but might not be contiguous.
    B = weight.t().contiguous()  # [K, N]
    # Strides (in elements)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)

    # Output
    C = torch.empty((M, N), device=x.device, dtype=torch.float32)

    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid: 2D over (M, N)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    scale = 1.0 + float(scaling_factor)

    matmul_bias_scale_kernel[grid](
        A, B, bias if bias is not None else tl.zeros((), dtype=tl.float32), C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    return C


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model.
    Fuses matmul + bias + scaling into a single kernel.

    Entry point name: ModelNew
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Reuse nn.Linear for parameter storage & initialization
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: y = (x @ W^T) * (1 + s) + b
        Uses Triton kernel on CUDA, falls back to torch on CPU.
        """
        # Fallback to torch if not CUDA
        if not x.is_cuda:
            # Pure PyTorch, but apply the algebraic simplification
            y = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
            return y * (1.0 + self.scaling_factor)

        # Ensure dtype is float32 for this kernel (can be extended to fp16/bf16)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        w = self.linear.weight
        b = self.linear.bias
        # Launch Triton
        y = _triton_linear(x, w, b, self.scaling_factor)
        return y


# The following helpers mirror your originals
batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
