"""BERT encoder-only model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..L2.bert_embeddings import BertEmbeddings
from .bert_encoder import BertEncoder


@dataclass
class EncoderModelOutput:
    last_hidden_state: torch.Tensor


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        return attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> EncoderModelOutput | tuple[torch.Tensor]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify input_ids or inputs_embeds")

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        extended_attention_mask = self._prepare_attention_mask(attention_mask, device)
        hidden_states = self.encoder(embedding_output, attention_mask=extended_attention_mask)

        if not return_dict:
            return (hidden_states,)
        return EncoderModelOutput(last_hidden_state=hidden_states)
