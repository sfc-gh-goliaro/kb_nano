"""RMSNorm with optional fused residual add.

Uses Triton kernels that correctly handle all element positions,
replacing sgl_kernel which has a stride bug zeroing even-indexed elements.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, out_ptr, w_ptr,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * N
    out_row = out_ptr + row.to(tl.int64) * N

    sum_sq = tl.zeros([], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        val = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(val * val)

    rrms = tl.math.rsqrt(sum_sq / N + eps)

    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        val = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        out = val * rrms * w
        tl.store(out_row + cols, out.to(out_row.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr,        # [M, N] in/out: receives normed result (in-place)
    res_ptr,      # [M, N] in/out: receives x + residual (in-place)
    w_ptr,        # [N] weights
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    """x, residual -> residual_out = x + residual, x_out = RMSNorm(residual_out)."""
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * N
    res_row = res_ptr + row.to(tl.int64) * N

    # Phase 1: add + accumulate variance
    sum_sq = tl.zeros([], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x_val = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        res_val = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        added = x_val + res_val
        tl.store(res_row + cols, added.to(res_row.dtype.element_ty), mask=mask)
        sum_sq += tl.sum(added * added)

    rrms = tl.math.rsqrt(sum_sq / N + eps)

    # Phase 2: normalize and store to x
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        added = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = added * rrms * w
        tl.store(x_row + cols, normed.to(x_row.dtype.element_ty), mask=mask)


def _rmsnorm(x, weight, eps):
    assert x.is_contiguous()
    M = x.shape[0] if x.ndim == 2 else x.numel() // x.shape[-1]
    N = x.shape[-1]
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(min(N, 1024))
    num_warps = min(max(BLOCK // 256, 1), 8)
    flat_x = x.view(-1, N)
    flat_out = out.view(-1, N)
    _rmsnorm_kernel[(M,)](
        flat_x, flat_out, weight,
        N=N, eps=eps, BLOCK=BLOCK,
        num_warps=num_warps, num_stages=1,
    )
    return out


def _fused_add_rmsnorm(x, residual, weight, eps):
    assert x.is_contiguous() and residual.is_contiguous()
    M = x.shape[0] if x.ndim == 2 else x.numel() // x.shape[-1]
    N = x.shape[-1]
    BLOCK = triton.next_power_of_2(min(N, 1024))
    num_warps = min(max(BLOCK // 256, 1), 8)
    flat_x = x.view(-1, N)
    flat_res = residual.view(-1, N)
    _fused_add_rmsnorm_kernel[(M,)](
        flat_x, flat_res, weight,
        N=N, eps=eps, BLOCK=BLOCK,
        num_warps=num_warps, num_stages=1,
    )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        if residual is None:
            return _rmsnorm(x, self.weight, self.eps)
        else:
            _fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
