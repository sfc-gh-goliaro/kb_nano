"""GPT-OSS decoder layer: attention + MoE with RMSNorm residual connections.

Uses the shared ``LlamaAttention`` with ``use_sinks=True`` and
``sliding_window`` to implement GPT-OSS attention sinks and per-layer
sliding window. Rotary embedding is passed through forward (created
once at the model level and shared across layers).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L1.allreduce import fused_allreduce_rmsnorm, has_flashinfer_ar_workspace
from ..L2.attention import LlamaAttention
from ..L2.gpt_oss_moe import GptOssMoE
from ....infra.tp import _tp_size


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            bias=True,
            o_proj_bias=True,
            use_sinks=True,
            sliding_window=config.sliding_window,
            layer_idx=layer_idx,
        )
        self.mlp = GptOssMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _use_compiled_fused_ar_norm() -> bool:
        return (
            torch.compiler.is_compiling()
            and _tp_size() > 1
            and has_flashinfer_ar_workspace()
        )

    @staticmethod
    def _fused_ar_norm(norm: RMSNorm, hidden_states, residual):
        residual_is_zero = residual is None
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return fused_allreduce_rmsnorm(
            hidden_states,
            residual,
            norm.weight,
            norm.eps,
            residual_is_zero=residual_is_zero,
        )

    def forward(self, positions, hidden_states, residual, rotary_emb):
        if self._use_compiled_fused_ar_norm():
            return self.forward_compiled_fused_ar_norm(
                positions, hidden_states, residual, rotary_emb,
                fuse_input_ar_norm=True,
                fuse_mlp_ar_norm=True,
            )

        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb=rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_compiled_fused_ar_norm(
        self,
        positions,
        hidden_states,
        residual,
        rotary_emb,
        *,
        fuse_input_ar_norm: bool,
        fuse_mlp_ar_norm: bool,
    ):
        if fuse_input_ar_norm:
            hidden_states, residual = self._fused_ar_norm(
                self.input_layernorm, hidden_states, residual,
            )
        elif residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn.forward_local_o_proj(
            positions, hidden_states, rotary_emb=rotary_emb,
        )
        hidden_states, residual = self._fused_ar_norm(
            self.post_attention_layernorm, hidden_states, residual,
        )
        if fuse_mlp_ar_norm:
            hidden_states = self.mlp.forward_local(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
