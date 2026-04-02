"""ColBERTv2 encoder model and MaxSim scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from ..L1.linear import Linear
from ..L2.bert_embeddings import BertEmbeddings
from ..L3.bert_layer import BertLayer


@dataclass
class ColBERTv2Config:
    model_type: str = "bert"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    dtype: torch.dtype = torch.bfloat16
    dim: int = 128
    query_maxlen: int = 32
    doc_maxlen: int = 220
    query_marker_token_id: int = 1
    doc_marker_token_id: int = 2
    mask_token_id: int = 103
    cls_token_id: int = 101
    sep_token_id: int = 102
    mask_punctuation: bool = True
    interaction: str = "colbert"
    similarity: str = "cosine"

    @classmethod
    def from_pretrained(cls, model_name: str) -> "ColBERTv2Config":
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return cls(
            model_type=getattr(hf, "model_type", "bert"),
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            intermediate_size=hf.intermediate_size,
            max_position_embeddings=hf.max_position_embeddings,
            type_vocab_size=hf.type_vocab_size,
            layer_norm_eps=hf.layer_norm_eps,
            hidden_act=hf.hidden_act,
            hidden_dropout_prob=hf.hidden_dropout_prob,
            attention_probs_dropout_prob=hf.attention_probs_dropout_prob,
            pad_token_id=hf.pad_token_id,
            position_embedding_type=getattr(hf, "position_embedding_type", "absolute"),
        )


class BertEncoder(nn.Module):
    def __init__(self, config: ColBERTv2Config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask=attention_mask)
        return hidden_states


class BertModel(nn.Module):
    def __init__(self, config: ColBERTv2Config):
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
    ) -> BaseModelOutput | tuple[torch.Tensor]:
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
        return BaseModelOutput(last_hidden_state=hidden_states)


class ColBERTv2ModelForInference(nn.Module):
    def __init__(self, config: ColBERTv2Config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.linear = Linear(config.hidden_size, config.dim, bias=False)
        self.pad_token = config.pad_token_id
        self.skiplist: set[int] = set()

    def set_skiplist(self, token_ids: set[int] | list[int] | tuple[int, ...]) -> None:
        self.skiplist = {int(token_id) for token_id in token_ids}

    def mask(
        self,
        input_ids: torch.Tensor,
        skiplist: set[int] | None = None,
    ) -> torch.Tensor:
        blocked = {self.pad_token}
        if skiplist:
            blocked |= {int(token_id) for token_id in skiplist}
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in blocked:
            mask &= input_ids.ne(token_id)
        return mask

    def query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        query_vecs = self.linear(hidden_states)
        query_mask = self.mask(input_ids, skiplist=set()).unsqueeze(-1).float()
        return F.normalize(query_vecs * query_mask, p=2, dim=2)

    def doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        doc_vecs = self.linear(hidden_states)
        doc_mask = self.mask(input_ids, skiplist=self.skiplist)
        doc_vecs = F.normalize(doc_vecs * doc_mask.unsqueeze(-1).float(), p=2, dim=2)
        if return_mask:
            return doc_vecs, doc_mask
        return doc_vecs

    def score(
        self,
        query_vecs: torch.Tensor,
        doc_vecs: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = doc_vecs @ query_vecs.to(dtype=doc_vecs.dtype).permute(0, 2, 1)
        return self.score_reduce(scores, doc_mask)

    def score_reduce(
        self,
        scores_padded: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        doc_padding = ~doc_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
        scores_padded = scores_padded.masked_fill(doc_padding.unsqueeze(-1), -9999)
        return scores_padded.max(1).values.sum(-1)

    def forward(
        self,
        query_input: tuple[torch.Tensor, torch.Tensor] | None = None,
        doc_input: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if query_input is None or doc_input is None:
            raise ValueError("query_input and doc_input must both be provided")
        query_vecs = self.query(*query_input)
        doc_vecs, doc_mask = self.doc(*doc_input, return_mask=True)
        return self.score(query_vecs, doc_vecs, doc_mask)
