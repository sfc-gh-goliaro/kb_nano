"""GLA SwiGLU MLP block: gate_proj + up_proj -> SiLU gate -> down_proj.

Uses separate linear layers (not merged) to match FLA checkpoint weight names.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
