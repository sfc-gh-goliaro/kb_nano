"""Kimi-Linear decoder layer: hybrid KDA/MLA attention + MoE/dense MLP (L3).

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
                rms_norm_eps=config.rms_norm_eps,
            )

        # MoE for sparse layers, dense SwiGLU MLP for layer 0.
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

    def forward(self, hidden_states, residual, layer_state=None):
        # sgl_kernel RMSNorm requires 2D; reshape around norm calls
        shape = hidden_states.shape
        h2d = hidden_states.reshape(-1, shape[-1])

        if residual is None:
            residual = h2d
            hidden_states = self.input_layernorm(h2d)
        else:
            residual = residual.reshape(-1, shape[-1])
            hidden_states, residual = self.input_layernorm(h2d, residual)

        hidden_states = hidden_states.reshape(shape)
        residual = residual.reshape(shape)

        # Both KDA and MLA share the (hidden_states, layer_state) signature.
        hidden_states = self.self_attn(hidden_states, layer_state=layer_state)

        shape = hidden_states.shape
        h2d = hidden_states.reshape(-1, shape[-1])
        r2d = residual.reshape(-1, shape[-1])
        hidden_states, residual = self.post_attention_layernorm(h2d, r2d)
        hidden_states = hidden_states.reshape(shape)
        residual = residual.reshape(shape)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
