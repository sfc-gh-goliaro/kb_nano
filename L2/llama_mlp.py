"""Llama SwiGLU MLP block: gate_up_proj -> SiluAndMul -> down_proj."""

from __future__ import annotations

import torch.nn as nn

from ..infra.tp import MergedColumnParallelLinear, RowParallelLinear
from ..L1.silu_and_mul import SiluAndMul


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)
