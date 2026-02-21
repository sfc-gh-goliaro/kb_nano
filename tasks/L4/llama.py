"""Standalone Llama 3.1 model implementation.

Uses Llama 3.1-style RoPE with frequency scaling, SwiGLU MLP,
and GQA attention. Supports tensor parallelism via shared TP layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L1.rms_norm import RMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L3.llama_decoder import LlamaDecoderLayer


@dataclass
class LlamaConfig:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 8.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position_embeddings: int = 8192
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "LlamaConfig":
        hf = AutoConfig.from_pretrained(model_name)
        rope = hf.rope_scaling or {}
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=hf.rope_theta,
            rope_scaling_factor=rope.get("factor", 1.0),
            rope_low_freq_factor=rope.get("low_freq_factor", 1.0),
            rope_high_freq_factor=rope.get("high_freq_factor", 1.0),
            rope_original_max_position_embeddings=rope.get(
                "original_max_position_embeddings", hf.max_position_embeddings,
            ),
        )


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_low_freq_factor=config.rope_low_freq_factor,
            rope_high_freq_factor=config.rope_high_freq_factor,
            rope_original_max_position_embeddings=config.rope_original_max_position_embeddings,
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, self.rotary_emb)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def forward_with_lm_proj(self, input_ids, positions):
        """Forward pass including LM head linear (no gather)."""
        hidden_states = self.model(input_ids, positions)
        return self.lm_head.project(hidden_states)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, partial_logits):
        """Gather and finalize logits from pre-computed local partition."""
        logits = self.lm_head.gather_logits(partial_logits)
        if logits is not None:
            logits = logits.float()
        return logits

    def greedy_sample_decode(self, partial_logits):
        """Fast greedy sampling path: local argmax + small allgather."""
        result = self.lm_head.gather_greedy(partial_logits.float())
        return result
