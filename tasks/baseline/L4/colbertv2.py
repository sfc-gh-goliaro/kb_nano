"""ColBERTv2 encoder model and MaxSim scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L2.colbertv2_ops import (
    ColBERTv2Embedding,
    ColBERTv2MaxSim,
    ColBERTv2ScoreReduce,
    ColBERTv2TokenMask,
)
from ..L3.bert_model import BertModel


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


class ColBERTv2ModelForInference(nn.Module):
    def __init__(self, config: ColBERTv2Config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.embedding = ColBERTv2Embedding(config.hidden_size, config.dim)
        self.token_mask = ColBERTv2TokenMask(config.pad_token_id)
        self.maxsim = ColBERTv2MaxSim()
        self.score_reducer = ColBERTv2ScoreReduce()
        self.skiplist: set[int] = set()

    def set_skiplist(self, token_ids: set[int] | list[int] | tuple[int, ...]) -> None:
        self.skiplist = {int(token_id) for token_id in token_ids}

    def mask(
        self,
        input_ids: torch.Tensor,
        skiplist: set[int] | None = None,
    ) -> torch.Tensor:
        return self.token_mask(input_ids, skiplist=skiplist)

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
        query_mask = self.mask(input_ids, skiplist=set())
        return self.embedding(hidden_states, query_mask)

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
        doc_mask = self.mask(input_ids, skiplist=self.skiplist)
        doc_vecs = self.embedding(hidden_states, doc_mask)
        if return_mask:
            return doc_vecs, doc_mask
        return doc_vecs

    def score(
        self,
        query_vecs: torch.Tensor,
        doc_vecs: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.maxsim(query_vecs, doc_vecs, doc_mask)

    def score_reduce(
        self,
        scores_padded: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_reducer(scores_padded, doc_mask)

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
