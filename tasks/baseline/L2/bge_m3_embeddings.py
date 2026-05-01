"""BGE-M3 embedding heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.l2_norm import L2Norm
from ..L1.linear import Linear
from ..L1.relu import ReLU


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


class BGEM3SparseEmbedding(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, pad_token_id: int | None):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.sparse_linear = Linear(hidden_size, 1, bias=True)
        self.relu = ReLU()

    def forward(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        return_embedding: bool = True,
    ) -> torch.Tensor:
        token_weights = self.relu(self.sparse_linear(hidden_state))
        if not return_embedding:
            return token_weights

        sparse_embedding = torch.zeros(
            input_ids.size(0),
            self.vocab_size,
            dtype=token_weights.dtype,
            device=token_weights.device,
        )
        sparse_embedding = sparse_embedding.scatter_reduce(
            dim=-1,
            index=input_ids,
            src=token_weights.squeeze(-1),
            reduce="amax",
        )
        unused_tokens = [
            token_id
            for token_id in (
                self.pad_token_id,
                0,
                2,
                3,
            )
            if token_id is not None
        ]
        sparse_embedding[:, unused_tokens] *= 0.0
        return sparse_embedding


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
