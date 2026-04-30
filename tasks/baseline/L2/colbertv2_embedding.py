"""ColBERTv2 contextual token projection."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.l2_norm import L2Norm
from ..L1.linear import Linear


class ColBERTv2Embedding(nn.Module):
    def __init__(self, hidden_size: int, dim: int):
        super().__init__()
        self.linear = Linear(hidden_size, dim, bias=False)
        self.norm = L2Norm(dim=2)

    def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        vecs = self.linear(hidden_states)
        return self.norm(vecs * token_mask.unsqueeze(-1).float())
