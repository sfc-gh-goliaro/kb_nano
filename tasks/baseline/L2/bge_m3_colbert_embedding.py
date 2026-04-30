"""BGE-M3 ColBERT token embedding head."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.l2_norm import L2Norm
from ..L1.linear import Linear


class BGEM3ColBERTEmbedding(nn.Module):
    def __init__(self, hidden_size: int, colbert_dim: int, normalize_embeddings: bool = True):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.colbert_linear = Linear(hidden_size, colbert_dim, bias=True)
        self.norm = L2Norm(dim=-1)

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * attention_mask[:, 1:][:, :, None].float()
        if self.normalize_embeddings:
            colbert_vecs = self.norm(colbert_vecs)
        return colbert_vecs
