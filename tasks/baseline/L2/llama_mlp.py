"""Llama SwiGLU MLP block: gate_up_proj -> SiluAndMul -> down_proj."""

from __future__ import annotations

import torch.nn as nn

from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear
from ..L1.silu_and_mul import SiluAndMul


class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
            quant_config=quant_config,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)
