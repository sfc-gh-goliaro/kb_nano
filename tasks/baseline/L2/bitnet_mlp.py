"""BitNet MLP: squared-ReLU gated FFN with W1.58A8 BitLinear and sub-norm.

Architecture (per ``microsoft/bitnet-b1.58-2B-4T``):

    x ─► BitLinear(gate), BitLinear(up) ─► relu(gate)^2 * up
       ─► RMSNorm ─► BitLinear(down)

Weight names match the HuggingFace checkpoint convention so that the shared
weight loader can populate them directly:

    mlp.gate_proj.weight  / .weight_scale
    mlp.up_proj.weight    / .weight_scale
    mlp.down_proj.weight  / .weight_scale
    mlp.ffn_sub_norm.weight
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.bitnet_linear import BitLinear
from ..L1.rms_norm import RMSNorm


class BitNetMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)
        self.ffn_sub_norm = RMSNorm(intermediate_size, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        inner = F.relu(gate).square() * up
        inner = self.ffn_sub_norm(inner)
        return self.down_proj(inner)
