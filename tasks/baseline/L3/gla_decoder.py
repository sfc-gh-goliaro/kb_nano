"""GLA decoder layer: attn_norm -> GLA attention -> residual,
                      mlp_norm -> SwiGLU MLP -> residual.

Pre-norm residual pattern matching the FLA architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.gla_attention import GatedLinearAttention
from ..L2.gla_mlp import GLAMLP


class GLADecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedLinearAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            norm_eps=config.norm_eps,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.attn_norm(
            hidden_states.reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.mlp_norm(
            hidden_states.reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
