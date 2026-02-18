"""
Standalone Llama 3.1 model implementation.

Uses Llama 3.1-style RoPE with frequency scaling, SwiGLU MLP,
and GQA attention. Supports tensor parallelism via shared TP layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..ops import (
    Attention,
    MergedColumnParallelLinear,
    ParallelLMHead,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    VocabParallelEmbedding,
    _apply_rotary_emb,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Llama 3.1 RoPE with frequency scaling
# ---------------------------------------------------------------------------
class Llama3RotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.head_size = config.head_dim
        inv_freq = self._compute_inv_freq(config)
        t = torch.arange(config.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, config):
        base, dim = config.rope_theta, config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        low_wl = config.rope_original_max_position_embeddings / config.rope_low_freq_factor
        high_wl = config.rope_original_max_position_embeddings / config.rope_high_freq_factor
        wl = 2 * math.pi / inv_freq
        if config.rope_low_freq_factor != config.rope_high_freq_factor:
            smooth = (config.rope_original_max_position_embeddings / wl
                      - config.rope_low_freq_factor) / (
                      config.rope_high_freq_factor - config.rope_low_freq_factor)
        else:
            smooth = torch.zeros_like(inv_freq)
        return torch.where(wl < high_wl, inv_freq,
                           torch.where(wl > low_wl, inv_freq / config.rope_scaling_factor,
                                       (1 - smooth) * inv_freq / config.rope_scaling_factor + smooth * inv_freq))

    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return _apply_rotary_emb(query, cos, sin), _apply_rotary_emb(key, cos, sin)


# ---------------------------------------------------------------------------
# Llama MLP
# ---------------------------------------------------------------------------
class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)


# ---------------------------------------------------------------------------
# Llama Decoder Layer
# ---------------------------------------------------------------------------
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Llama3RotaryEmbedding(config)

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

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits
