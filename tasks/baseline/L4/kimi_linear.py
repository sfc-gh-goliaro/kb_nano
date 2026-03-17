"""Standalone Kimi-Linear model implementation.

Matches moonshotai/Kimi-Linear-48B-A3B-Instruct architecture:
    - Hybrid: 20 KDA (Delta-Net) layers + 7 MLA (latent attention) layers
    - MoE: 256 routed experts (top-8, sigmoid), 1 shared expert
    - Layer 0: dense MLP; layers 1-26: MoE
    - hidden_size=2304, 27 layers, vocab=163840

Weight names match HuggingFace checkpoint convention:
    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.*
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.block_sparse_moe.*  (layers 1-26)
    model.layers.{i}.mlp.*               (layer 0)
    model.norm.weight
    lm_head.weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.rms_norm import RMSNorm
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L3.kimi_linear_decoder import KimiLinearDecoderLayer


@dataclass
class KimiLinearConfig:
    hidden_size: int = 2304
    num_hidden_layers: int = 27
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 72
    intermediate_size: int = 9216
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    # MLA
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # MoE
    num_experts: int = 256
    num_experts_per_token: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 1024
    moe_layer_freq: int = 1
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 2.446
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    # KDA / linear attention
    kda_layers: list[int] = field(default_factory=lambda: [
        1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26
    ])
    full_attn_layers: list[int] = field(default_factory=lambda: [
        4, 8, 12, 16, 20, 24, 27
    ])
    kda_num_heads: int = 32
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    # Runtime
    model_type: str = "kimi_linear"
    dtype: torch.dtype = torch.bfloat16

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Check if layer uses KDA attention (1-indexed in config)."""
        return (layer_idx + 1) in self.kda_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.num_experts is not None
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "KimiLinearConfig":
        try:
            hf = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        except Exception:
            path = Path(model_name_or_path) / "config.json"
            with path.open() as f:
                data = json.load(f)
            return cls._from_dict(data)
        return cls._from_hf(hf)

    @classmethod
    def _from_hf(cls, hf) -> "KimiLinearConfig":
        kda_config = getattr(hf, "linear_attn_config", None) or {}
        rope_params = getattr(hf, "rope_parameters", None) or {}
        rope_theta = rope_params.get("rope_theta", getattr(hf, "rope_theta", 10000.0))
        return cls(
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            intermediate_size=hf.intermediate_size,
            vocab_size=hf.vocab_size,
            rms_norm_eps=getattr(hf, "rms_norm_eps", 1e-5),
            rope_theta=rope_theta,
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
            kv_lora_rank=getattr(hf, "kv_lora_rank", 512),
            q_lora_rank=getattr(hf, "q_lora_rank", None),
            qk_nope_head_dim=getattr(hf, "qk_nope_head_dim", 128),
            qk_rope_head_dim=getattr(hf, "qk_rope_head_dim", 64),
            v_head_dim=getattr(hf, "v_head_dim", 128),
            num_experts=getattr(hf, "num_experts", 256),
            num_experts_per_token=getattr(hf, "num_experts_per_token", 8),
            num_shared_experts=getattr(hf, "num_shared_experts", 1),
            moe_intermediate_size=getattr(hf, "moe_intermediate_size", 1024),
            moe_layer_freq=getattr(hf, "moe_layer_freq", 1),
            moe_renormalize=getattr(hf, "moe_renormalize", True),
            moe_router_activation_func=getattr(hf, "moe_router_activation_func", "sigmoid"),
            routed_scaling_factor=getattr(hf, "routed_scaling_factor", 2.446),
            use_grouped_topk=getattr(hf, "use_grouped_topk", True),
            num_expert_group=getattr(hf, "num_expert_group", 1),
            topk_group=getattr(hf, "topk_group", 1),
            first_k_dense_replace=getattr(hf, "first_k_dense_replace", 1),
            kda_layers=kda_config.get("kda_layers", []),
            full_attn_layers=kda_config.get("full_attn_layers", []),
            kda_num_heads=kda_config.get("num_heads", 32),
            kda_head_dim=kda_config.get("head_dim", 128),
            short_conv_kernel_size=kda_config.get("short_conv_kernel_size", 4),
        )

    @classmethod
    def _from_dict(cls, data: dict) -> "KimiLinearConfig":
        kda_config = data.get("linear_attn_config", {}) or {}
        rope_params = data.get("rope_parameters", {}) or {}
        rope_theta = rope_params.get("rope_theta", data.get("rope_theta", 10000.0))
        return cls(
            hidden_size=data.get("hidden_size", 2304),
            num_hidden_layers=data.get("num_hidden_layers", 27),
            num_attention_heads=data.get("num_attention_heads", 32),
            num_key_value_heads=data.get("num_key_value_heads", 32),
            head_dim=data.get("head_dim", 72),
            intermediate_size=data.get("intermediate_size", 9216),
            vocab_size=data.get("vocab_size", 163840),
            rms_norm_eps=data.get("rms_norm_eps", 1e-5),
            rope_theta=rope_theta,
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            kv_lora_rank=data.get("kv_lora_rank", 512),
            q_lora_rank=data.get("q_lora_rank", None),
            qk_nope_head_dim=data.get("qk_nope_head_dim", 128),
            qk_rope_head_dim=data.get("qk_rope_head_dim", 64),
            v_head_dim=data.get("v_head_dim", 128),
            num_experts=data.get("num_experts", 256),
            num_experts_per_token=data.get("num_experts_per_token", 8),
            num_shared_experts=data.get("num_shared_experts", 1),
            moe_intermediate_size=data.get("moe_intermediate_size", 1024),
            moe_layer_freq=data.get("moe_layer_freq", 1),
            moe_renormalize=data.get("moe_renormalize", True),
            moe_router_activation_func=data.get("moe_router_activation_func", "sigmoid"),
            routed_scaling_factor=data.get("routed_scaling_factor", 2.446),
            use_grouped_topk=data.get("use_grouped_topk", True),
            num_expert_group=data.get("num_expert_group", 1),
            topk_group=data.get("topk_group", 1),
            first_k_dense_replace=data.get("first_k_dense_replace", 1),
            kda_layers=kda_config.get("kda_layers", []),
            full_attn_layers=kda_config.get("full_attn_layers", []),
            kda_num_heads=kda_config.get("num_heads", 32),
            kda_head_dim=kda_config.get("head_dim", 128),
            short_conv_kernel_size=kda_config.get("short_conv_kernel_size", 4),
        )


class KimiLinearModel(nn.Module):
    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [KimiLinearDecoderLayer(config, layer_idx=i)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        residual = None
        for i, layer in enumerate(self.layers):
            layer_state = None
            if past_key_values is not None:
                layer_state = past_key_values[i]
            hidden_states, residual = layer(
                hidden_states, residual, layer_state=layer_state,
            )

        # Final norm (sgl_kernel requires 2D)
        shape = hidden_states.shape
        h2d = hidden_states.reshape(-1, shape[-1])
        r2d = residual.reshape(-1, shape[-1])
        hidden_states, _ = self.norm(h2d, r2d)
        return hidden_states.reshape(shape)


class KimiLinearForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.model = KimiLinearModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions=None, past_key_values=None, use_cache=False):
        return self.model(input_ids, positions, past_key_values, use_cache)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, hidden_states):
        return self.compute_logits(hidden_states)
