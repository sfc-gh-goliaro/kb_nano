"""BGE-M3 encoder model for dense, sparse, and ColBERT outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig

from ..L2.bge_m3_embeddings import (
    BGEM3ColBERTEmbedding,
    BGEM3DenseEmbedding,
    BGEM3SparseEmbedding,
)
from ..L3.xlm_roberta_model import XLMRobertaModel


@dataclass
class BGEM3Config:
    model_type: str = "xlm-roberta"
    vocab_size: int = 250002
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 8194
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    pad_token_id: int = 1
    position_embedding_type: str = "absolute"
    dtype: torch.dtype = torch.bfloat16
    colbert_dim: int = 1024

    @classmethod
    def from_pretrained(cls, model_name: str) -> "BGEM3Config":
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return cls(
            model_type=getattr(hf, "model_type", "xlm-roberta"),
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
            colbert_dim=hf.hidden_size,
        )


class BGEM3ModelForInference(XLMRobertaModel):
    """FlagEmbedding-aligned BGE-M3 inference wrapper."""

    def __init__(
        self,
        config: BGEM3Config,
        sentence_pooling_method: str = "cls",
        normalize_embeddings: bool = True,
    ):
        super().__init__(config)
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.dense_embedding = BGEM3DenseEmbedding(
            sentence_pooling_method=sentence_pooling_method,
            normalize_embeddings=normalize_embeddings,
        )
        self.sparse_embedding = BGEM3SparseEmbedding(
            config.hidden_size,
            config.vocab_size,
            config.pad_token_id,
        )
        self.colbert_embedding = BGEM3ColBERTEmbedding(
            config.hidden_size,
            config.colbert_dim,
            normalize_embeddings=normalize_embeddings,
        )

    def forward(
        self,
        text_input: dict[str, torch.Tensor] | None = None,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        return_sparse_embedding: bool = False,
    ) -> dict[str, torch.Tensor]:
        if text_input is None:
            raise ValueError("text_input must be provided")
        if not (return_dense or return_sparse or return_colbert_vecs):
            raise ValueError("At least one output mode must be enabled")

        outputs = super().forward(return_dict=True, **text_input)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = text_input["attention_mask"]

        result: dict[str, torch.Tensor] = {}
        if return_dense:
            result["dense_vecs"] = self.dense_embedding(last_hidden_state, attention_mask)
        if return_sparse:
            result["sparse_vecs"] = self.sparse_embedding(
                last_hidden_state,
                text_input["input_ids"],
                return_embedding=return_sparse_embedding,
            )
        if return_colbert_vecs:
            result["colbert_vecs"] = self.colbert_embedding(last_hidden_state, attention_mask)
        return result
