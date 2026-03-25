"""SiLU-and-Mul activation: silu(x) * y where [x, y] = chunk(input, 2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel import silu_and_mul as _sgl_silu_and_mul


class _ActivationBuffer:
    """Mutable container so multiple SiluAndMul layers can share one buffer.

    When multiple layers share the same buffer and request different column
    counts, ``b[:rows, :cols]`` can be non-contiguous (stride[0] != cols).
    sgl_kernel's silu_and_mul assumes a contiguous output, so we fall back
    to a separate contiguous buffer for the smaller-column case.
    """
    __slots__ = ("buf", "_small")

    def __init__(self):
        self.buf: torch.Tensor | None = None
        self._small: torch.Tensor | None = None

    def get(self, rows: int, cols: int, device, dtype) -> torch.Tensor:
        b = self.buf
        if b is None or b.size(0) < rows or b.size(1) < cols:
            self.buf = b = torch.empty(rows, cols, device=device, dtype=dtype)
        if cols == b.size(1):
            return b[:rows]
        s = self._small
        if s is None or s.size(0) < rows or s.size(1) != cols:
            self._small = s = torch.empty(rows, cols, device=device, dtype=dtype)
        return s[:rows]


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        self._act_buf = _ActivationBuffer()

    def set_shared_buffer(self, shared: _ActivationBuffer):
        self._act_buf = shared

    def forward(self, x):
        half = x.size(-1) // 2
        out = self._act_buf.get(x.size(0), half, x.device, x.dtype)
        return _sgl_silu_and_mul(x, out)
