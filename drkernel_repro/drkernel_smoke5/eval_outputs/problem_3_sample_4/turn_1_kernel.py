import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Fused GEMM + bias + scale:
# C = alpha * (A @ B) + beta
# A: [M, K], B: [K, N], C: [M, N]
@triton.jit
def gemm_bias_scale_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers for blocks
    A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # promote to fp32 for dot
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)
        # advance
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # add bias and scale
    # bias shape [N], broadcast over M
    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
    bias = bias.to(tl.float32)
    acc = acc * alpha + bias[None, :]

    # store
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original model.
    Computes: out = (x @ W^T + b) * (1 + scaling_factor)
    Fused into a single kernel: out = alpha * (x @ W^T) + beta, where
    alpha = 1 + s, beta = b * (1 + s).
    Falls back to torch if Triton/CUDA not available.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        # reuse nn.Linear to hold parameters (initialization, state dict compatibility)
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU fallback or no triton: use high-level fused expression
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            alpha = 1.0 + self.scaling_factor
            # pre-scale bias; F.linear fuses bias add inside cuBLASLt
            if self.linear.bias is not None:
                bias_scaled = self.linear.bias * alpha
            else:
                bias_scaled = None
            return torch.nn.functional.linear(x, self.linear.weight, bias_scaled)

        # CUDA + Triton path: fused kernel
        W = self.linear.weight            # [N, K]
        b = self.linear.bias              # [N] or None
        assert W.is_cuda and x.is_cuda, "Inputs must be CUDA tensors for Triton path."

        M, K = x.shape
        N = W.shape[0]
        assert W.shape[1] == K, f"Weight shape mismatch: got W {W.shape} for x.shape {x.shape}"

        # Allocate output
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = x.stride(0)
        stride_ak = x.stride(1)

        # View W as B[k, n] = W[n, k] without copy by using strides
        # W is [N, K]; we want [K, N] access pattern
        # stride over "k" dim: W.stride(1)
        # stride over "n" dim: W.stride(0)
        stride_bk = W.stride(1)
        stride_bn = W.stride(0)

        stride_cm = out.stride(0)
        stride_cn = out.stride(1)

        # Bias: ensure float32 and on device
        if b is not None:
            bias = b.to(dtype=torch.float32, device=x.device)
        else:
            # create a zero bias to satisfy kernel signature (will be added but is zero)
            bias = torch.zeros((N,), dtype=torch.float32, device=x.device)

        # Alpha and beta
        alpha = 1.0 + self.scaling_factor
        beta = bias * alpha  # pre-fused: out = alpha*A@B + beta; here beta = b*alpha

        # Choose block sizes
        # For these shapes, 128 is fine; you can tune 64/128/256.
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        gemm_bias_scale_kernel[grid](
            x, W, beta, out,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            alpha, bias,  # beta points to bias*alpha; pass bias to satisfy signature (unused)
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )
        return out


        # Fallback: If you need autograd and training support:
        if x.requires_grad or y.requires_grad:
            raise RuntimeError("Triton kernel is not diff-friendly (no backward)")
        return out


class ModelNew(nn.Module):
    def __init__(self, kernel: triton.jit):
        super().__init__()
        def triton_impl(self, x, w, y):
            # simple elementwise sum kernel: c[i] = a[i] + b[i]
            # If we want to replicate PyTorch behavior exactly (including gradients),
            # we should implement backward for x and y. For elementwise ops, PyTorch has efficient kernels already.
            pass
        