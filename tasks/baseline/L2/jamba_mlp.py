"""Jamba's plain (non-MoE) FFN block: SwiGLU.

Reference: ``transformers.models.jamba.modeling_jamba.JambaMLP``.

Implementation matches Llama-style SwiGLU exactly:

    down(silu(gate(x)) * up(x))

Layer-uniform with :class:`JambaMoE`: both expose ``forward(x) -> y``
with the same shape so the L3 decoder can switch between them based on
``layers_num_experts[layer_idx]``.

L1 ops: ``Linear``, ``SiLU``.  No ``F.linear`` / ``F.silu`` — those are
abstracted behind L1 modules per the L2 strict-L1-only rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU


class JambaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
