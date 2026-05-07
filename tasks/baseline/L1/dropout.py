"""Dropout wrapping F.dropout.

Accepts ``inplace`` kwarg for HF drop-in compatibility (some HF models pass
``nn.Dropout(p, inplace=True)``). The forward output is identical regardless
of inplace; ``inplace=True`` only avoids an allocation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    def __init__(self, p: float = 0.0, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=self.training, inplace=self.inplace)
