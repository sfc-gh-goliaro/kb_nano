"""Per-token-group FP8 activation quantization.

Quantizes BF16 activations to float8_e4m3fn with per-group scale factors.
Uses vLLM's CUDA kernel when available, falls back to Triton.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max  # 448.0
_FP8_MIN = -_FP8_MAX

_HAS_VLLM_CUDA_QUANT = False
try:
    from vllm import _custom_ops  # noqa: F401
    _HAS_VLLM_CUDA_QUANT = (hasattr(torch.ops, "_C")
                             and hasattr(torch.ops._C, "per_token_group_fp8_quant"))
except ImportError:
    pass


@triton.jit
def _per_token_group_quant_fp8_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


class PerTokenGroupQuantFP8(nn.Module):
    """Quantize BF16 activations to FP8 with per-token-group scaling.

    Args:
        group_size: Number of elements per quantization group (default 128).

    forward(x) -> (x_q, x_s):
        x:   [*, K] in BF16/FP16 (K must be divisible by group_size)
        x_q: [*, K] in float8_e4m3fn
        x_s: [*, K // group_size] in float32
    """

    def __init__(self, group_size: int = 128):
        super().__init__()
        self.group_size = group_size
        self._q_buf = None
        self._s_buf = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] % self.group_size == 0
        assert x.is_contiguous()

        q_shape = x.shape
        s_shape = x.shape[:-1] + (x.shape[-1] // self.group_size,)
        n = x.numel()
        s_n = 1
        for d in s_shape:
            s_n *= d

        if self._q_buf is None or self._q_buf.numel() < n:
            self._q_buf = torch.empty(n, device=x.device, dtype=_FP8_DTYPE)
        if self._s_buf is None or self._s_buf.numel() < s_n:
            self._s_buf = torch.empty(s_n, device=x.device, dtype=torch.float32)

        x_q = self._q_buf[:n].view(q_shape)
        x_s = self._s_buf[:s_n].view(s_shape)

        if _HAS_VLLM_CUDA_QUANT:
            torch.ops._C.per_token_group_fp8_quant(
                x, x_q, x_s,
                self.group_size, 1e-10,
                _FP8_MIN, _FP8_MAX, False,
            )
            return x_q, x_s

        M = x.numel() // self.group_size
        N = self.group_size
        BLOCK = triton.next_power_of_2(N)
        num_warps = min(max(BLOCK // 256, 1), 8)

        _per_token_group_quant_fp8_kernel[(M,)](
            x, x_q, x_s,
            self.group_size,
            x.shape[-1],
            x.stride(-2),
            1e-10,
            fp8_min=_FP8_MIN,
            fp8_max=_FP8_MAX,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )

        return x_q, x_s
