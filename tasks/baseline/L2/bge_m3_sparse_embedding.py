"""BGE-M3 lexical sparse embedding head."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.relu import ReLU


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
