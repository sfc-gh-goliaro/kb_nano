"""YOLOv10 Distribution Focal Loss layer."""

from __future__ import annotations

from pathlib import Path
import sys
_L2_DIR = Path(__file__).resolve().parent
_L1_DIR = _L2_DIR.parent / "L1"
for _p in (str(_L2_DIR), str(_L1_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import torch
import torch.nn as nn

from conv2d import Conv2d
from softmax import Softmax


class YOLODFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(self._softmax(x.view(b, 4, self.c1, a).transpose(2, 1))).view(b, 4, a)
