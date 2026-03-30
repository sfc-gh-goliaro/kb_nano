"""SiLU-and-Mul activation: silu(x) * y where [x, y] = chunk(input, 2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .csrc import _C


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
