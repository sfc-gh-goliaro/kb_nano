"""Vision MLP for Qwen vision transformer blocks.

Unified across Qwen2-VL (QuickGELU) and Qwen3-VL (SiLU) activations.
"""

from __future__ import annotations

from pathlib import Path
import sys
_L2_DIR = Path(__file__).resolve().parent
_L1_DIR = _L2_DIR.parent / "L1"
for _p in (str(_L2_DIR), str(_L1_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from collections.abc import Callable

import torch
import torch.nn as nn

from quickgelu import QuickGELU
from parallel_linear import ColumnParallelLinear, RowParallelLinear

class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))
