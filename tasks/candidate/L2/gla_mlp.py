"""SwiGLU MLP for GLA / RetNet decoder layers.

Three-projection variant matching FLA's checkpoint format:
  ``gate_proj.weight`` / ``up_proj.weight`` / ``down_proj.weight``

The existing ``L2.swiglu_mlp.SwiGLUMlp`` uses a different parameter
naming scheme (``fc1_g`` / ``fc1_x`` / ``fc2``), so we keep this thin
FLA-named variant rather than remapping checkpoint keys at load time.

Built exclusively from L1 ops.
"""

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

from linear import Linear
from silu import SiLU


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
