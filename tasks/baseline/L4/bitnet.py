"""Standalone BitNet b1.58 model implementation.

Matches the Microsoft BitNet-b1.58-2B-4T architecture:
    - GQA (20 query heads, 5 KV heads, head_dim=128)
    - RoPE (theta=500000)
    - Squared ReLU activation (not SwiGLU)
    - Per-token int8 activation quantization in all projections
    - Attention and FFN sub-norms
    - Tied embeddings (lm_head shares weight with embed_tokens)

Weight names match HuggingFace checkpoint convention:
    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.self_attn.attn_sub_norm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.layers.{i}.mlp.ffn_sub_norm.weight
    model.norm.weight
    lm_head.weight  (tied with model.embed_tokens.weight)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..L3.bitnet_decoder import BitNetBlock


@dataclass
class BitNetConfig:
    hidden_size: int = 2560
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    head_dim: int = 128
    intermediate_size: int = 6912
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_dict(cls, data: dict) -> "BitNetConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values() if f.name != "dtype"}
        kwargs = {k: data[k] for k in keys if k in data}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "BitNetConfig":
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class BitNetModel(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [BitNetBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions=None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class BitNetForCausalLM(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.model = BitNetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions=None) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)
