"""Jamba decoder layer wiring.

Reference: ``transformers.models.jamba.modeling_jamba``
            (``JambaAttentionDecoderLayer`` and ``JambaMambaDecoderLayer``).

Each Jamba layer is one of two flavours:

  * **Attention layer**  -- pre-norm + multi-head attention + pre-FF
                            norm + (MLP or sparse MoE).
  * **Mamba layer**      -- pre-norm + Mamba mixer + pre-FF norm
                            + (MLP or sparse MoE).

The attn/Mamba choice and the MLP-vs-MoE choice are encoded in the
config:

    config.layers_block_type[layer_idx] in {"attention", "mamba"}
    config.layers_num_experts[layer_idx]   # 1 -> dense MLP, >1 -> MoE

For ``ai21labs/Jamba-tiny-dev`` and ``ai21labs/Jamba-v0.1`` the recipe is
periodic:
  * Attention layers at indices ``i % attn_layer_period == attn_layer_offset``
    (period 8, offset 4 -- one attn per 8-layer block).
  * MoE layers at indices ``i % expert_layer_period == expert_layer_offset``
    (period 2, offset 1 -- every other layer is sparse).

Residual convention (matches HuggingFace's reference exactly):

    h0 = h
    h  = norm1(h)
    h  = mixer(h)            # mamba or attention
    h  = h + h0

    h0 = h
    h  = norm2(h)
    h  = ffn(h)              # mlp or moe
    h  = h + h0

We keep the standard ``[B, T, hidden]`` layout end-to-end.

L1 ops used: ``RMSNorm`` (full hidden_size, multiple of 32, safe path).
L2 ops used: ``JambaAttention``, ``JambaMambaMixer``, ``JambaMLP``,
             ``JambaMoE``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.jamba_attention import JambaAttention
from ..L2.jamba_mamba_mixer import JambaMambaMixer
from ..L2.jamba_mlp import JambaMLP
from ..L2.jamba_moe import JambaMoE


def _make_feed_forward(config, num_experts: int) -> nn.Module:
    if num_experts > 1:
        return JambaMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
        )
    return JambaMLP(config.hidden_size, config.intermediate_size)


class JambaAttentionDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mamba_layer = False
        self.self_attn = JambaAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )
        num_experts = config.layers_num_experts[layer_idx]
        self.feed_forward = _make_feed_forward(config, num_experts)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        cache_writeback: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_slab: tuple[torch.Tensor, torch.Tensor] | None = None,
        slot_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_k, new_v = self.self_attn(
            hidden_states,
            past_key=past_key, past_value=past_value,
            attention_mask=attention_mask,
            cache_writeback=cache_writeback,
            kv_slab=kv_slab,
            slot_pos=slot_pos,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_k, new_v


class JambaMambaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mamba_layer = True
        self.mamba = JambaMambaMixer(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=config.mamba_expand * config.hidden_size,
            time_step_rank=config.mamba_dt_rank,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            rms_norm_eps=config.rms_norm_eps,
            layer_idx=layer_idx,
        )
        num_experts = config.layers_num_experts[layer_idx]
        self.feed_forward = _make_feed_forward(config, num_experts)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,                 # [B, T, hidden]
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        is_decode: bool,
        mamba_pad_mask_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # The mixer operates on a flat ``[total_tokens, hidden]`` layout
        # so we can call into the vLLM Mamba kernels directly.  Reshape
        # in and out around the mixer call.
        b, t, d = hidden_states.shape
        flat = hidden_states.reshape(b * t, d)
        flat = self.mamba(
            flat,
            conv_state=conv_state,
            ssm_state=ssm_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            has_initial_state=has_initial_state,
            is_decode=is_decode,
            mamba_pad_mask=mamba_pad_mask_flat,
        )
        hidden_states = flat.reshape(b, t, d)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
