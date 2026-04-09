"""GELUTanh-and-Mul activation (Gemma-style): gelu_tanh(gate) * up.

Counterpart to ``SiluAndMul`` for models using GELU with tanh approximation
(e.g. Gemma / PaliGemma).  The input is a concatenation of [gate, up] along
the last dimension; the output is ``gelu_tanh(gate) * up``.

Currently implemented as pure PyTorch only (no custom CUDA kernel), which
allows torch.compile / Inductor to fuse it with adjacent ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]
