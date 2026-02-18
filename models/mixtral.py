"""
Standalone Mixtral-8x7B model implementation.

Adds Mixture-of-Experts (MoE) with top-k gating on top of the shared
attention and normalization layers. Uses standard RoPE (no frequency scaling).
Supports tensor parallelism via shared TP layers.

The MoE layer uses a fused Triton grouped-GEMM kernel for high throughput.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from ..ops import (
    Attention,
    ParallelLMHead,
    RMSNorm,
    RotaryEmbedding,
    VocabParallelEmbedding,
    _tp_rank,
    _tp_size,
    fused_experts,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MixtralConfig:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 32000
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MixtralConfig":
        hf = AutoConfig.from_pretrained(model_name)
        head_dim = getattr(hf, "head_dim", None)
        if head_dim is None:
            head_dim = hf.hidden_size // hf.num_attention_heads
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=hf.rope_theta,
            num_local_experts=hf.num_local_experts,
            num_experts_per_tok=hf.num_experts_per_tok,
        )


# ---------------------------------------------------------------------------
# MoE layer — fused Triton grouped GEMM
# ---------------------------------------------------------------------------
class MixtralMoE(nn.Module):
    """Mixture-of-Experts with fused Triton grouped GEMM.

    Weights:
      w13: [E, 2*intermediate_per_tp, hidden_size] — gate (w1) and up (w3) stacked
      w2:  [E, hidden_size, intermediate_per_tp]
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp

        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.gate.weight.weight_loader = lambda p, w: p.data.copy_(w)

        self.w13 = nn.Parameter(torch.empty(
            config.num_local_experts, 2 * self.intermediate_per_tp, config.hidden_size,
        ))
        self.w13.weight_loader = self._w13_weight_loader

        self.w2 = nn.Parameter(torch.empty(
            config.num_local_experts, config.hidden_size, self.intermediate_per_tp,
        ))
        self.w2.weight_loader = self._w2_weight_loader

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        """Load w1 or w3 into the stacked w13 parameter."""
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        """Load w2 (row-parallel): shard input dim."""
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        out = fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
        )

        if self.tp_size > 1:
            dist.all_reduce(out)

        return out.view(orig_shape)


# ---------------------------------------------------------------------------
# Mixtral Decoder Layer
# ---------------------------------------------------------------------------
class MixtralDecoderLayer(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
        )
        self.block_sparse_moe = MixtralMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class MixtralModel(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MixtralDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, self.rotary_emb)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.model = MixtralModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits
