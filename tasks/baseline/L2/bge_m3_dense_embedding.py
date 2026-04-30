"""BGE-M3 dense sentence embedding head."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.l2_norm import L2Norm


class BGEM3DenseEmbedding(nn.Module):
    def __init__(self, sentence_pooling_method: str = "cls", normalize_embeddings: bool = True):
        super().__init__()
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.norm = L2Norm(dim=-1)

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.sentence_pooling_method == "cls":
            dense_vecs = last_hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            summed = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            denom = attention_mask.sum(dim=1, keepdim=True).float()
            dense_vecs = summed / denom
        elif self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                dense_vecs = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch = last_hidden_state.shape[0]
                dense_vecs = last_hidden_state[
                    torch.arange(batch, device=last_hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(
                f"Unsupported pooling method: {self.sentence_pooling_method}",
            )
        if self.normalize_embeddings:
            dense_vecs = self.norm(dense_vecs)
        return dense_vecs
