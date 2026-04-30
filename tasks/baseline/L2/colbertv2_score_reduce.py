"""ColBERT score tensor reduction."""

from __future__ import annotations

import torch
import torch.nn as nn


class ColBERTv2ScoreReduce(nn.Module):
    def forward(
        self,
        scores_padded: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        doc_padding = ~doc_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
        scores_padded = scores_padded.masked_fill(doc_padding.unsqueeze(-1), -9999)
        return scores_padded.max(1).values.sum(-1)
