"""Standalone RWKV7 model implementation.

Matches FLA checkpoint weight names exactly:
  model.embeddings.weight
  model.layers.0.pre_norm.{weight,bias}
  model.layers.{i}.attn_norm.{weight,bias}
  model.layers.{i}.attn.x_{r,w,k,v,a,g}
  model.layers.{i}.attn.{r,k,v,o}_proj.weight
  model.layers.{i}.attn.k_k, k_a, r_k
  model.layers.{i}.attn.{w,a,g}_lora.lora.{0,2}.{weight,bias}
  model.layers.{i}.attn.v_lora.lora.{0,2}.{weight,bias}  (layers > 0)
  model.layers.{i}.attn.g_norm.{weight,bias}
  model.layers.{i}.ffn_norm.{weight,bias}
  model.layers.{i}.ffn.x_k
  model.layers.{i}.ffn.{key,value}.weight
  model.norm.{weight,bias}
  lm_head.weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from ..L3.rwkv7_decoder import RWKV7Block


@dataclass
class RWKV7Config:
    hidden_size: int = 2560
    head_dim: int = 64
    num_heads: int | None = None
    num_hidden_layers: int = 32
    vocab_size: int = 65536
    decay_low_rank_dim: int = 96
    gate_low_rank_dim: int = 320
    a_low_rank_dim: int = 96
    v_low_rank_dim: int = 64
    hidden_ratio: float = 4.0
    intermediate_size: int | None = None
    norm_bias: bool = True
    norm_eps: float = 1e-5
    norm_first: bool = True
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.num_heads is None:
            self.num_heads = self.hidden_size // self.head_dim
        if self.intermediate_size is None:
            intermediate = int(self.hidden_size * self.hidden_ratio)
            self.intermediate_size = 32 * ((intermediate + 31) // 32)

    @classmethod
    def from_dict(cls, data: dict) -> "RWKV7Config":
        keys = {f.name for f in cls.__dataclass_fields__.values() if f.name != "dtype"}
        kwargs = {k: data[k] for k in keys if k in data}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "RWKV7Config":
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class RWKV7Model(nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [RWKV7Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, bias=config.norm_bias, eps=config.norm_eps,
        )

    def forward(self, input_ids: torch.Tensor, positions=None) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)

        v_first = torch.zeros_like(hidden_states)
        for layer in self.layers:
            hidden_states, v_first = layer(hidden_states, v_first)

        return self.norm(hidden_states)


class RWKV7ForCausalLM(nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config
        self.model = RWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.weight

    def forward(self, input_ids: torch.Tensor, positions=None) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)
