"""ColBERT MaxSim score reduction."""

from __future__ import annotations

import torch
import torch.nn as nn

from .colbertv2_score_reduce import ColBERTv2ScoreReduce


class ColBERTv2MaxSim(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_reduce = ColBERTv2ScoreReduce()

    def forward(
        self,
        query_vecs: torch.Tensor,
        doc_vecs: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = doc_vecs @ query_vecs.to(dtype=doc_vecs.dtype).permute(0, 2, 1)
        return self.score_reduce(scores, doc_mask)
