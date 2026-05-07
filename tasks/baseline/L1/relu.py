"""ReLU activation: max(0, x).

Wraps ``F.relu``. Accepts the ``inplace`` kwarg of ``torch.nn.ReLU`` so HF
models that pass ``nn.ReLU(inplace=True)`` (regnet, sam_hq, etc.) drop in
without modification. Inference-time behavior is identical regardless of
inplace; ``inplace=True`` only avoids an allocation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, inplace=self.inplace)
