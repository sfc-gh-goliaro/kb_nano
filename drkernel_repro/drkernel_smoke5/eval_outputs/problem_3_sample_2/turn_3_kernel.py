import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def matmul_bias_scale_kernel(
    A,         # [M, K]
    BT,        # [K, N] contiguous
    Bias,      # [N] or None (we'll pass a valid pointer and guard loads)
    C,         # [M, N]
    M, N, K,
    stride_am, stride_ak,     # A strides
    stride_bk, stride_bn,     # BT strides
    stride_cm, stride_cn,     # C strides
    scale,                    # scalar float: 1 + scaling_factor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first K-block
    A_ptr = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # [BM, BK]
    BT_ptr = BT + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # [BK, BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in steps of BLOCK_K
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        a = tl.load(A_ptr, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(BT_ptr, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

        A_ptr += BLOCK_K * stride_ak
        BT_ptr += BLOCK_K * stride_bk

    # Epilogue: add bias, then scale
    # Load bias for the current column block if provided
    # We assume Bias is a valid pointer to [N]; if Bias is None in Python, we should not pass it.
    bias_vals = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
    c = acc + bias_vals[None, :]          # add bias
    c = c * scale                         # scale

    # Store
    C_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptr, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _triton_linear(x: torch.Tensor,
                   weight: torch.Tensor,
                   bias: torch.Tensor,
                   scaling_factor: float) -> torch.Tensor:
    """
    Compute y = (x @ weight.T) * (1 + scaling_factor) + bias
    using a single fused Triton kernel.

    Args:
        x: [M, K]
        weight: [N, K] as in nn.Linear
        bias: [N] or None
        scaling_factor: float
    Returns:
        y: [M, N]
    """
    assert x.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    if bias is not None:
        assert bias.is_cuda, "Bias must be CUDA"

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, f"weight shape mismatch: got {weight.shape}, expected (*, {K})"

    # Use a contiguous transpose for clean strides and coalesced access
    BT = weight.t().contiguous()  # [K, N]

    # Strides in elements
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    stride_bk = BT.stride(0)
    stride_bn = BT.stride(1)

    # Output (float32 to match inputs by default)
    C = torch.empty((M, N), device=x.device, dtype=torch.float32)

    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    scale = 1.0 + float(scaling_factor)

    matmul_bias_scale_kernel[grid](
        x, BT, bias,  # pass real bias tensor
        C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return C


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model.
    Fuses matmul + bias + scaling into a single kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            # CPU fallback with simplification
            y = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
            return y * (1.0 + self.scaling_factor)

        w = self.linear.weight
        b = self.linear.bias

        y = _triton_linear(x, w, b, self.scaling_factor)
        return y


# Defaults
batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
