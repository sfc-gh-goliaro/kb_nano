"""Standalone RetNet (Retentive Network) model implementation.

Matches FLA checkpoint weight names exactly:
  model.embeddings.weight
  model.layers.{i}.attn_norm.weight
  model.layers.{i}.attn.{q,k,v,g,o}_proj.weight
  model.layers.{i}.attn.g_norm_swish_gate.weight
  model.layers.{i}.mlp_norm.weight
  model.layers.{i}.mlp.{gate,up,down}_proj.weight
  model.norm.weight
  lm_head.weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L3.retnet_decoder import RetNetDecoderLayer


@dataclass
class RetNetConfig:
    hidden_size: int = 2560
    num_heads: int = 10
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    expand_k: float = 1.0
    expand_v: float = 2.0
    hidden_ratio: int = 2
    intermediate_size: int | None = None
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.intermediate_size is None:
            intermediate = int(self.hidden_size * self.hidden_ratio * 2 / 3)
            self.intermediate_size = 256 * ((intermediate + 255) // 256)

    @classmethod
    def from_dict(cls, data: dict) -> "RetNetConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values() if f.name != "dtype"}
        kwargs = {k: data[k] for k in keys if k in data}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "RetNetConfig":
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class RetNetModel(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [RetNetDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, input_ids: torch.Tensor, positions=None) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        norm_dtype = self.norm.weight.dtype
        return self.norm(
            hidden_states.to(dtype=norm_dtype).reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)


class RetNetForCausalLM(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        self.model = RetNetModel(config)
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
