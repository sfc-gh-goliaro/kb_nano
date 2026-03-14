"""Mamba2 model implementation (Codestral / SSD architecture).

Matches HuggingFace checkpoint weight names exactly:
  backbone.embeddings.weight                    [vocab_size, hidden_size]
  backbone.layers.{i}.norm.weight               [hidden_size]
  backbone.layers.{i}.mixer.in_proj.weight      [in_proj_size, hidden_size]
  backbone.layers.{i}.mixer.conv1d.weight       [conv_dim, 1, conv_kernel]
  backbone.layers.{i}.mixer.conv1d.bias         [conv_dim]
  backbone.layers.{i}.mixer.A_log               [num_heads]
  backbone.layers.{i}.mixer.D                   [num_heads]
  backbone.layers.{i}.mixer.dt_bias             [num_heads]
  backbone.layers.{i}.mixer.norm.weight         [intermediate_size]
  backbone.layers.{i}.mixer.out_proj.weight     [hidden_size, intermediate_size]
  backbone.norm_f.weight                        [hidden_size]
  lm_head.weight                                [vocab_size, hidden_size]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L3.mamba2_decoder import Mamba2DecoderLayer


@dataclass
class Mamba2Config:
    model_type: str = "mamba2"
    hidden_size: int = 4096
    num_hidden_layers: int = 64
    intermediate_size: int = 8192
    state_size: int = 128
    conv_kernel: int = 4
    expand: int = 2
    n_groups: int = 8
    num_heads: int = 128
    head_dim: int = 64
    chunk_size: int = 256
    vocab_size: int = 32768
    use_bias: bool = False
    use_conv_bias: bool = True
    tie_word_embeddings: bool = False
    layer_norm_epsilon: float = 1e-5
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "Mamba2Config":
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        hidden_size = data.get("hidden_size", 4096)
        expand = data.get("expand", 2)
        num_heads = data.get("num_heads", 128)
        head_dim = data.get("head_dim", 64)
        intermediate_size = data.get("intermediate_size", num_heads * head_dim)

        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=data.get("num_hidden_layers", data.get("n_layer", 64)),
            intermediate_size=intermediate_size,
            state_size=data.get("state_size", 128),
            conv_kernel=data.get("conv_kernel", 4),
            expand=expand,
            n_groups=data.get("n_groups", 8),
            num_heads=num_heads,
            head_dim=head_dim,
            chunk_size=data.get("chunk_size", 256),
            vocab_size=data.get("vocab_size", 32768),
            use_bias=data.get("use_bias", False),
            use_conv_bias=data.get("use_conv_bias", True),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            layer_norm_epsilon=data.get("layer_norm_epsilon", 1e-5),
        )


class Mamba2Model(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Mamba2DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, cache_params=None, cache_position=None):
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cache_params=cache_params, cache_position=cache_position,
            )
        # Final norm (sgl_kernel requires 2D)
        shape = hidden_states.shape
        hidden_states = self.norm_f(hidden_states.reshape(-1, shape[-1]))
        return hidden_states.reshape(shape)


class Mamba2ForCausalLM(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.backbone = Mamba2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.weight

    def forward(self, input_ids, cache_params=None, cache_position=None):
        return self.backbone(
            input_ids, cache_params=cache_params, cache_position=cache_position,
        )

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, hidden_states):
        return self.compute_logits(hidden_states)
