"""Kimi-Linear decoder layer: hybrid KDA/MLA attention + MoE/dense MLP (L3).

Operates on a flat varlen batch ``[num_actual_tokens, hidden_size]``
throughout (no reshape gymnastics around residuals or norms): the engine
packs all sequences into a single 2D activation and the layer threads it
through KDA or MLA + MoE/MLP without ever materializing a 3D ``[B, T, D]``
tensor.

Dispatches to KDA (Delta-Net) or MLA (latent attention) based on layer
index. Uses MoE for most layers, dense SwiGLU MLP for layer 0.
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.kda_attention import KDAAttention
from ..L2.mla_attention import MLAAttention
from ..L2.shared_expert_moe import SharedExpertMoE, _TPSwiGLUMLP


class KimiLinearDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)

        if self.is_kda:
            self.self_attn = KDAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.kda_num_heads,
                head_dim=config.kda_head_dim,
                layer_idx=layer_idx,
                conv_kernel_size=config.short_conv_kernel_size,
                rms_norm_eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = MLAAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                kv_lora_rank=config.kv_lora_rank,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
            )

        if config.is_moe_layer(layer_idx):
            self.block_sparse_moe = SharedExpertMoE(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_token,
                moe_intermediate_size=config.moe_intermediate_size,
                routing="sigmoid",
                correction_bias=True,
                renormalize=config.moe_renormalize,
                routed_scaling_factor=config.routed_scaling_factor,
                shared_expert_intermediate_size=(
                    config.moe_intermediate_size * config.num_shared_experts
                ),
                shared_expert_attr_name="shared_experts",
                shared_expert_gate=False,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = _TPSwiGLUMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, residual, state_manager=None):
        # hidden_states: [num_actual_tokens, hidden_size] (already flat 2D)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, state_manager=state_manager)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
