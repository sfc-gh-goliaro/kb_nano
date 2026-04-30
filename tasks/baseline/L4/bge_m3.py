"""BGE-M3 encoder model for dense, sparse, and ColBERT outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from ..L1.relu import ReLU
from ..L1.linear import Linear
from ..L2.xlm_roberta_embeddings import XLMRobertaEmbeddings
from ..L3.xlm_roberta_layer import XLMRobertaLayer


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


class XLMRobertaEncoder(nn.Module):
    def __init__(self, config: BGEM3Config):
        super().__init__()
        self.layer = nn.ModuleList([
            XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask=attention_mask)
        return hidden_states


class XLMRobertaModel(nn.Module):
    def __init__(self, config: BGEM3Config):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)

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
        self.sparse_linear = Linear(config.hidden_size, 1, bias=True)
        self.colbert_linear = Linear(config.hidden_size, config.colbert_dim, bias=True)
        self.relu = ReLU()

    def _dense_embedding(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        if self.sentence_pooling_method == "mean":
            summed = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            denom = attention_mask.sum(dim=1, keepdim=True).float()
            return summed / denom
        if self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch, device=last_hidden_state.device),
                sequence_lengths,
            ]
        raise NotImplementedError(
            f"Unsupported pooling method: {self.sentence_pooling_method}",
        )

    def _sparse_embedding(
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
            self.config.vocab_size,
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
                self.config.pad_token_id,
                0,  # <s>
                2,  # </s>
                3,  # <unk>
            )
            if token_id is not None
        ]
        sparse_embedding[:, unused_tokens] *= 0.0
        return sparse_embedding

    def _colbert_embedding(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        return colbert_vecs * attention_mask[:, 1:][:, :, None].float()

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
            dense_vecs = self._dense_embedding(last_hidden_state, attention_mask)
            if self.normalize_embeddings:
                dense_vecs = F.normalize(dense_vecs, dim=-1)
            result["dense_vecs"] = dense_vecs
        if return_sparse:
            result["sparse_vecs"] = self._sparse_embedding(
                last_hidden_state,
                text_input["input_ids"],
                return_embedding=return_sparse_embedding,
            )
        if return_colbert_vecs:
            colbert_vecs = self._colbert_embedding(last_hidden_state, attention_mask)
            if self.normalize_embeddings:
                colbert_vecs = F.normalize(colbert_vecs, dim=-1)
            result["colbert_vecs"] = colbert_vecs
        return result
