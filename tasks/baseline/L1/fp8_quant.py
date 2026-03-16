"""Per-token-group FP8 activation quantization.

Quantizes BF16 activations to float8_e4m3fn with per-group scale factors.
On Blackwell with DeepGEMM E8M0, uses vLLM's packed UE8M0 quantization
for exact numerical match. Otherwise uses sgl_kernel, vLLM CUDA, or Triton.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max  # 448.0
_FP8_MIN = -_FP8_MAX

# Check if E8M0 packed quantization is available (Blackwell + DeepGEMM)
_HAS_PACKED_E8M0 = False
_packed_quant = None
try:
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
    if is_deep_gemm_e8m0_used():
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8_packed_for_deepgemm as _packed_quant,
        )
        _HAS_PACKED_E8M0 = True
except (ImportError, AssertionError):
    pass

_HAS_SGL_QUANT = False
if not _HAS_PACKED_E8M0:
    try:
        from sgl_kernel import sgl_per_token_group_quant_fp8 as _sgl_quant
        # Some sgl_kernel versions require sglang at runtime; do a real call.
        _test_x = torch.ones(1, 128, dtype=torch.bfloat16, device="cuda")
        _test_q = torch.empty(1, 128, dtype=_FP8_DTYPE, device="cuda")
        _test_s = torch.empty(1, 1, dtype=torch.float32, device="cuda")
        _sgl_quant(_test_x, _test_q, _test_s, 128, 1e-10, _FP8_MIN, _FP8_MAX)
        del _test_x, _test_q, _test_s
        _HAS_SGL_QUANT = True
    except Exception:
        pass

_HAS_VLLM_CUDA_QUANT = False
if not _HAS_PACKED_E8M0 and not _HAS_SGL_QUANT:
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

    _shared_q_buf: torch.Tensor | None = None
    _shared_s_buf: torch.Tensor | None = None
    _use_shared = False

    def __init__(self, group_size: int = 128, use_packed_e8m0: bool = True):
        super().__init__()
        self.group_size = group_size
        self._use_packed_e8m0 = use_packed_e8m0 and _HAS_PACKED_E8M0
        self._q_buf = None
        self._s_buf = None

    @property
    def _q(self):
        return PerTokenGroupQuantFP8._shared_q_buf if PerTokenGroupQuantFP8._use_shared else self._q_buf

    @_q.setter
    def _q(self, val):
        if PerTokenGroupQuantFP8._use_shared:
            PerTokenGroupQuantFP8._shared_q_buf = val
        else:
            self._q_buf = val

    @property
    def _s(self):
        return PerTokenGroupQuantFP8._shared_s_buf if PerTokenGroupQuantFP8._use_shared else self._s_buf

    @_s.setter
    def _s(self, val):
        if PerTokenGroupQuantFP8._use_shared:
            PerTokenGroupQuantFP8._shared_s_buf = val
        else:
            self._s_buf = val

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] % self.group_size == 0
        assert x.is_contiguous()

        if self._use_packed_e8m0:
            return _packed_quant(x, group_size=self.group_size)

        q_shape = x.shape
        s_shape = x.shape[:-1] + (x.shape[-1] // self.group_size,)
        n = x.numel()
        s_n = 1
        for d in s_shape:
            s_n *= d

        q_buf = self._q
        s_buf = self._s
        if q_buf is None or q_buf.numel() < n:
            q_buf = torch.empty(n, device=x.device, dtype=_FP8_DTYPE)
            self._q = q_buf
        if s_buf is None or s_buf.numel() < s_n:
            s_buf = torch.empty(s_n, device=x.device, dtype=torch.float32)
            self._s = s_buf

        x_q = q_buf[:n].view(q_shape)
        x_s = s_buf[:s_n].view(s_shape)

        if _HAS_SGL_QUANT:
            _sgl_quant(x, x_q, x_s, self.group_size, 1e-10,
                       _FP8_MIN, _FP8_MAX)
            return x_q, x_s

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
