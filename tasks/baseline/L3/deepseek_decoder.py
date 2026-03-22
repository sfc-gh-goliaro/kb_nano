"""DeepSeek V3 decoder layer: MLA attention + MoE/MLP + optional DSA Indexer.

Dense MLP for first_k_dense_replace layers, MoE for the rest
(controlled by first_k_dense_replace and moe_layer_freq from config).

For V3.2 (when config has index_topk), includes a DeepSeekIndexer that
computes sparse attention indices before MLA attention.
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.deepseek_mla import DeepSeekMLA
from ..L2.deepseek_moe import DeepSeekMoE
from ..L2.llama_mlp import LlamaMLP


class _DenseMlpConfig:
    """Minimal config for LlamaMLP on dense layers."""
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size


class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, rotary_emb: nn.Module,
                 quant_config: dict | None = None,
                 topk_indices_buffer=None,
                 indexer_rotary_emb=None):
        super().__init__()
        self.has_indexer = hasattr(config, 'index_topk') and config.index_topk is not None
        self._topk_indices_buffer = topk_indices_buffer

        self.self_attn = DeepSeekMLA(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rms_norm_eps=config.rms_norm_eps,
            rotary_emb=rotary_emb,
            attn_scaling=config.attn_scaling,
            quant_config=quant_config,
        )

        if self.has_indexer and topk_indices_buffer is not None:
            from ..L2.deepseek_indexer import DeepSeekIndexer
            indexer = DeepSeekIndexer(
                hidden_size=config.hidden_size,
                q_lora_rank=config.q_lora_rank,
                index_n_heads=config.index_n_heads,
                index_head_dim=config.index_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                index_topk=config.index_topk,
                rms_norm_eps=config.rms_norm_eps,
                rotary_emb=indexer_rotary_emb,
                quant_config=quant_config,
            )
            self.self_attn.set_indexer(indexer)
            self.self_attn.set_topk_indices_buffer(topk_indices_buffer)

        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        moe_freq = getattr(config, "moe_layer_freq", 1)
        is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= first_k_dense
            and (moe_freq > 0 and layer_idx % moe_freq == 0)
        )

        if is_moe:
            self.mlp = DeepSeekMoE(config, quant_config=quant_config)
        else:
            dense_cfg = _DenseMlpConfig(
                config.hidden_size, config.intermediate_size,
            )
            self.mlp = LlamaMLP(dense_cfg, quant_config=quant_config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
