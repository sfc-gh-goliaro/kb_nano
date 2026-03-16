"""Fused RMSNorm + FP8 group quantization.

Combines residual-add, RMSNorm, and per-token-group FP8 quantization into
a single Triton kernel, eliminating intermediate BF16 writes between
the separate RMSNorm and FP8 quant kernels.

The residual-add variant follows the same memory convention as
_fused_add_rmsnorm_kernel in rms_norm.py: the updated residual
(x + old_residual) is written to the *residual* buffer, leaving x
untouched. This avoids aliasing issues when x points to a shared
GEMM output buffer.

Performance: uses a large BLOCK for the variance accumulation pass and
group_size-width blocks for the normalize+quantize pass, avoiding the
excessive loop overhead of iterating in group_size chunks during phase 1.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_FP8_MIN = -_FP8_MAX

_USE_UE8M0 = (torch.cuda.is_available()
              and torch.cuda.get_device_capability(0)[0] >= 10)


@triton.jit
def _fused_add_rmsnorm_fp8_kernel(
    x_ptr,          # [M, N] input (BF16), read-only
    residual_ptr,   # [M, N] residual (BF16), updated in-place to x+residual
    weight_ptr,     # [N] RMSNorm weight (FP32)
    out_fp8_ptr,    # [M, N] FP8 output
    out_scale_ptr,  # [M, N//group_size] FP32 scales
    N: tl.constexpr,
    eps,
    group_size: tl.constexpr,
    fp8_min,
    fp8_max,
    USE_UE8M0: tl.constexpr,
    BLOCK_VAR: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """One row: residual += x, then RMSNorm(residual) → FP8."""
    row = tl.program_id(0)
    row_offset = row.to(tl.int64) * N

    x_row = x_ptr + row_offset
    res_row = residual_ptr + row_offset

    groups_per_row = N // group_size

    # Phase 1: residual add (write to residual_ptr) + compute variance
    sum_sq = tl.zeros([], dtype=tl.float32)
    for start in range(0, N, BLOCK_VAR):
        cols = start + tl.arange(0, BLOCK_VAR)
        mask = cols < N
        x_val = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        res_val = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        added = x_val + res_val
        tl.store(res_row + cols, added.to(res_row.dtype.element_ty), mask=mask)
        sum_sq += tl.sum(added * added)

    rrms = tl.math.rsqrt(sum_sq / N + eps)

    # Phase 2: RMSNorm + FP8 quantize per group
    fp8_row = out_fp8_ptr + row.to(tl.int64) * N
    scale_row = out_scale_ptr + row.to(tl.int64) * groups_per_row

    for g in range(0, groups_per_row):
        g_start = g * group_size
        cols = g_start + tl.arange(0, BLOCK_G)
        full_mask = cols < (g_start + group_size)

        added = tl.load(res_row + cols, mask=full_mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=full_mask, other=1.0).to(tl.float32)
        normed = added * rrms * w

        _absmax = tl.maximum(tl.max(tl.abs(normed)), eps)
        scale_raw = _absmax / fp8_max
        scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if USE_UE8M0 else scale_raw
        quantized = tl.clamp(normed / scale, fp8_min, fp8_max).to(fp8_row.dtype.element_ty)

        tl.store(fp8_row + cols, quantized, mask=full_mask)
        tl.store(scale_row + g, scale)


@triton.jit
def _fused_rmsnorm_fp8_kernel(
    x_ptr,          # [M, N] input (BF16), read-only
    weight_ptr,     # [N] RMSNorm weight (FP32)
    out_fp8_ptr,    # [M, N] FP8 output
    out_scale_ptr,  # [M, N//group_size] FP32 scales
    N: tl.constexpr,
    eps,
    group_size: tl.constexpr,
    fp8_min,
    fp8_max,
    USE_UE8M0: tl.constexpr,
    BLOCK_VAR: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """RMSNorm + FP8 quant (no residual add). For the first layer."""
    row = tl.program_id(0)
    row_offset = row.to(tl.int64) * N
    x_row = x_ptr + row_offset
    groups_per_row = N // group_size

    sum_sq = tl.zeros([], dtype=tl.float32)
    for start in range(0, N, BLOCK_VAR):
        cols = start + tl.arange(0, BLOCK_VAR)
        mask = cols < N
        x_val = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_val * x_val)

    rrms = tl.math.rsqrt(sum_sq / N + eps)

    fp8_row = out_fp8_ptr + row.to(tl.int64) * N
    scale_row = out_scale_ptr + row.to(tl.int64) * groups_per_row

    for g in range(0, groups_per_row):
        g_start = g * group_size
        cols = g_start + tl.arange(0, BLOCK_G)
        full_mask = cols < (g_start + group_size)

        x_val = tl.load(x_row + cols, mask=full_mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=full_mask, other=1.0).to(tl.float32)
        normed = x_val * rrms * w

        _absmax = tl.maximum(tl.max(tl.abs(normed)), eps)
        scale_raw = _absmax / fp8_max
        scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if USE_UE8M0 else scale_raw
        quantized = tl.clamp(normed / scale, fp8_min, fp8_max).to(fp8_row.dtype.element_ty)

        tl.store(fp8_row + cols, quantized, mask=full_mask)
        tl.store(scale_row + g, scale)


class FusedRMSNormFP8Quant(nn.Module):
    """Fused residual-add + RMSNorm + FP8 group quantization.

    forward(x, residual=None) -> (fp8_out, fp8_scale, residual_out):
        x:            [M, N] BF16 input
        residual:     [M, N] BF16 residual (None for first layer)
        fp8_out:      [M, N] float8_e4m3fn
        fp8_scale:    [M, N//group_size] float32
        residual_out: [M, N] BF16

    When residual is provided, the kernel writes x+residual into the
    residual buffer (matching the unfused RMSNorm convention) and returns
    it. This is safe even when x aliases a shared GEMM output buffer.

    When residual is None (first layer), x is used directly and returned
    as the residual output.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 group_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self._fp8_buf = None
        self._scale_buf = None

    def _ensure_buffers(self, M, device):
        N = self.hidden_size
        n_fp8 = M * N
        n_scale = M * (N // self.group_size)
        if self._fp8_buf is None or self._fp8_buf.numel() < n_fp8:
            self._fp8_buf = torch.empty(n_fp8, device=device, dtype=_FP8_DTYPE)
        if self._scale_buf is None or self._scale_buf.numel() < n_scale:
            self._scale_buf = torch.empty(n_scale, device=device, dtype=torch.float32)

    def set_shared_buffers(self, fp8_buf, scale_buf):
        self._fp8_buf = fp8_buf
        self._scale_buf = scale_buf

    def forward(self, x, residual=None):
        M = x.shape[0]
        N = self.hidden_size
        assert x.shape[-1] == N
        assert x.is_contiguous()

        self._ensure_buffers(M, x.device)
        fp8_out = self._fp8_buf[:M * N].view(M, N)
        scale_out = self._scale_buf[:M * (N // self.group_size)].view(
            M, N // self.group_size)

        BLOCK_G = triton.next_power_of_2(self.group_size)
        BLOCK_VAR = triton.next_power_of_2(min(N, 1024))
        num_warps = min(max(BLOCK_VAR // 256, 1), 8)

        if residual is not None:
            assert residual.shape == x.shape
            _fused_add_rmsnorm_fp8_kernel[(M,)](
                x, residual, self.weight,
                fp8_out, scale_out,
                N=N,
                eps=self.eps,
                group_size=self.group_size,
                fp8_min=_FP8_MIN,
                fp8_max=_FP8_MAX,
                USE_UE8M0=_USE_UE8M0,
                BLOCK_VAR=BLOCK_VAR,
                BLOCK_G=BLOCK_G,
                num_warps=num_warps,
                num_stages=1,
            )
            return fp8_out, scale_out, residual
        else:
            _fused_rmsnorm_fp8_kernel[(M,)](
                x, self.weight,
                fp8_out, scale_out,
                N=N,
                eps=self.eps,
                group_size=self.group_size,
                fp8_min=_FP8_MIN,
                fp8_max=_FP8_MAX,
                USE_UE8M0=_USE_UE8M0,
                BLOCK_VAR=BLOCK_VAR,
                BLOCK_G=BLOCK_G,
                num_warps=num_warps,
                num_stages=1,
            )
            return fp8_out, scale_out, x
