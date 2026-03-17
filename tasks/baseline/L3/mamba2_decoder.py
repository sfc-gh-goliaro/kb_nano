"""Mamba2 decoder layer: RMSNorm + Mamba2Mixer with residual connection.

Weight names match HuggingFace checkpoint:
  layers.{i}.norm.weight    [hidden_size]
  layers.{i}.mixer.*        (see mamba2_mixer.py)
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.mamba2_mixer import Mamba2Mixer


class Mamba2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.mixer = Mamba2Mixer(config, layer_idx)
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, cache_params=None, cache_position=None):
        residual = hidden_states
        shape = hidden_states.shape
        # sgl_kernel RMSNorm requires 2D
        hidden_states = self.norm(hidden_states.reshape(-1, shape[-1]))
        hidden_states = hidden_states.reshape(shape)
        hidden_states = self.mixer(
            hidden_states, cache_params=cache_params, cache_position=cache_position,
        )
        return hidden_states + residual
