"""Shared token/position/type embedding wiring for encoder models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.embedding import Embedding
from ..L1.layer_norm import LayerNorm


def create_roberta_position_ids_from_input_ids(
    input_ids: torch.Tensor,
    padding_idx: int,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    mask = input_ids.ne(padding_idx).int()
    incremental = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental.long() + padding_idx


class EncoderEmbeddingsBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        word_padding_idx = self._word_embedding_padding_idx(config)
        position_padding_idx = self._position_embedding_padding_idx(config)

        self.word_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=word_padding_idx,
        )
        self.position_embeddings = Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=position_padding_idx,
        )
        self.token_type_embeddings = Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.LayerNorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def _word_embedding_padding_idx(self, config) -> int | None:
        return None

    def _position_embedding_padding_idx(self, config) -> int | None:
        return None

    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        past_key_values_length: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            if inputs_embeds is None:
                raise ValueError("inputs_embeds must be provided when input_ids is None")
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        seq_len = input_shape[1]
        if position_ids is None:
            position_ids = self._resolve_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

        if token_type_ids is None:
            buffered = self.token_type_ids[:, :seq_len]
            token_type_ids = buffered.expand(input_shape[0], seq_len)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = (
            inputs_embeds
            + self.token_type_embeddings(token_type_ids.to(device=device))
        )
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids.to(device=device))
        return self.LayerNorm(embeddings)
