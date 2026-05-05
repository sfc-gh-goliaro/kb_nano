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

Forward signature follows the project convention (matches Llama, Mamba,
Mamba2, Mixtral, ...):

    forward(self, positions, hidden_states, residual) -> (hidden_states, residual)

Per-step KV / Mamba state is read from the global ``Context`` populated
by the engine via ``set_jamba_context``; the L2 mixers
(``JambaAttention``, ``JambaMambaMixer``) reach into the context for
their per-layer slab using ``self.layer_idx``.

Residual handling.  Jamba's HF reference uses *non-fused* pre-norm:

    h0 = h
    h  = norm1(h);   h = mixer(h);   h = h + h0
    h0 = h
    h  = norm2(h);   h = ffn(h);     h = h + h0

(rather than Llama's add-then-RMS fused residual).  We honour that
exact arithmetic but expose the same ``(positions, h, residual)``
signature as the rest of the project so the L4 wiring stays uniform.
The convention here: if ``residual`` is non-None, we add it into
``hidden_states`` first (closing out the previous layer's pending
residual), then run the two Jamba sub-blocks in their natural form,
returning ``(new_hidden, None)`` -- residual is fully closed every
layer.

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
            layer_idx=layer_idx,
        )
        num_experts = config.layers_num_experts[layer_idx]
        self.feed_forward = _make_feed_forward(config, num_experts)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor | None,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        # Close out a pending residual from a prior layer (Llama-style
        # delayed add).  Jamba itself uses non-fused pre-norm, so we
        # eagerly fold any incoming residual back into ``hidden_states``.
        if residual is not None:
            hidden_states = hidden_states + residual

        # Attention block: pre-norm, mixer, residual add.
        h0 = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = h0 + hidden_states

        # FFN block: pre-norm, MLP / MoE, residual add.
        h0 = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = h0 + hidden_states
        return hidden_states, None


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
        positions: torch.Tensor | None,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        # Close out any pending residual (see JambaAttentionDecoderLayer).
        if residual is not None:
            hidden_states = hidden_states + residual

        # Mamba block: pre-norm, mixer, residual add.
        h0 = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(positions, hidden_states)
        hidden_states = h0 + hidden_states

        # FFN block: pre-norm, MLP / MoE, residual add.
        h0 = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = h0 + hidden_states
        return hidden_states, None
