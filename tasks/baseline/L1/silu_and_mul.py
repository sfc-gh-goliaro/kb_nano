"""SiLU-and-Mul activation: silu(x) * y where [x, y] = chunk(input, 2)."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load as _load_ext

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_C = _load_ext(
    name="kb_nano_L1_ops",
    sources=[os.path.join(_CSRC, f) for f in [
        "binding.cpp", "rmsnorm.cu", "activation.cu", "pos_enc.cu",
        "moe_sum.cu", "moe_align.cu", "moe_topk_softmax.cu",
    ]],
    extra_cuda_cflags=["-O3", "--use_fast_math",
                       "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16"],
    extra_cflags=["-O3"],
    verbose=False,
)


class _ActivationBuffer:
    """Mutable container so multiple SiluAndMul layers can share one buffer."""
    __slots__ = ("buf",)

    def __init__(self):
        self.buf: torch.Tensor | None = None

    def get(self, rows: int, cols: int, device, dtype) -> torch.Tensor:
        b = self.buf
        if b is None or b.size(0) < rows or b.size(1) < cols:
            self.buf = b = torch.empty(rows, cols, device=device, dtype=dtype)
        return b[:rows, :cols]


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        self._act_buf = _ActivationBuffer()

    def set_shared_buffer(self, shared: _ActivationBuffer):
        self._act_buf = shared

    def forward(self, x):
        half = x.size(-1) // 2
        out = self._act_buf.get(x.size(0), half, x.device, x.dtype)
        _C.silu_and_mul(out, x)
        return out
