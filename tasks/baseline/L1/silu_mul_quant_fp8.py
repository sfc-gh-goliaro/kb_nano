"""Fused SiLU+Mul+FP8 quantization kernel.

Single Triton kernel that applies SiLU activation, element-wise multiply, and
per-token-group FP8 quantization with UE8M0 (power-of-two) scales. Replaces
three separate kernel launches (SiLU+Mul, gather, FP8 quant) with one.

Matches vllm's ``silu_mul_per_token_group_quant_fp8_colmajor``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
_GROUP_SIZE: tl.constexpr = 128


@triton.jit
def _silu_mul_quant_fp8_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    M,
    N,
    y_s_col_stride: tl.int64,
    fp8_max: tl.constexpr,
    fp8_min: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset
    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    act_in = act_in.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (act_in / (one_f32 + tl.exp(-act_in))).to(y_ptr.dtype.element_ty)
    y = (silu_out * mul_in).to(tl.float32)

    eps = tl.cast(1e-12, tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)


class SiluMulQuantFp8(nn.Module):
    """Fused SiLU+Mul+FP8 quantization.

    Takes gate_up output [M, 2*N] and produces FP8 intermediate [M, N]
    with per-group UE8M0 scales, in a single kernel launch.
    """

    def forward(
        self,
        gate_up: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gate_up: [M, 2*N] bf16/fp16 — concatenated gate and up projections.

        Returns:
            (fp8_out, scales): fp8_out is [M, N] float8_e4m3fn,
                               scales is [M, N // 128] float32.
        """
        assert gate_up.ndim == 2
        M, N_full = gate_up.shape
        N = N_full // 2

        fp8_out = torch.empty(M, N, dtype=torch.float8_e4m3fn,
                              device=gate_up.device)
        scales = torch.empty(
            (N // _GROUP_SIZE, M), dtype=torch.float32, device=gate_up.device,
        ).transpose(0, 1)

        BLOCK_M = 8
        BLOCK_N = _GROUP_SIZE
        grid = (triton.cdiv(M, BLOCK_M), N // BLOCK_N)

        _silu_mul_quant_fp8_kernel[grid](
            gate_up,
            fp8_out,
            scales,
            M, N_full,
            scales.stride(-1),
            fp8_max=_FP8_INFO.max,
            fp8_min=_FP8_INFO.min,
            GROUP_SIZE=_GROUP_SIZE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return fp8_out, scales
