"""SiLU-and-Mul activation: silu(x) * y where [x, y] = chunk(input, 2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel import silu_and_mul as _sgl_silu_and_mul


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        self._out = None

    def forward(self, x):
        half = x.size(-1) // 2
        if self._out is None or self._out.size(0) < x.size(0) or self._out.size(-1) < half:
            self._out = torch.empty(
                x.size(0), half, device=x.device, dtype=x.dtype,
            )
        out = self._out[:x.size(0), :half]
        return _sgl_silu_and_mul(x, out)
