"""BitNet MLP: squared-ReLU gated FFN with sub-norm.

Architecture:
    x → BitLinear(gate), BitLinear(up) → relu²(gate) * up
      → RMSNorm(sub) → BitLinear(down)

Weight names match HuggingFace checkpoint convention:
    mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    mlp.ffn_sub_norm.weight
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.bitnet_linear import BitLinear


class BitNetMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)
        self.ffn_sub_norm = nn.RMSNorm(intermediate_size, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        inner = F.relu(gate).square() * up
        inner = self.ffn_sub_norm(inner)
        return self.down_proj(inner)
