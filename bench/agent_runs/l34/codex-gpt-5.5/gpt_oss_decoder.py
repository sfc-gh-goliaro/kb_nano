from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.gpt_oss_moe import GptOssMoE


_HAS_VLLM_RMS = (
    hasattr(torch.ops, "_C")
    and hasattr(torch.ops._C, "rms_norm")
    and hasattr(torch.ops._C, "fused_add_rms_norm")
)
_HAS_FK_RMS = (
    hasattr(torch.ops, "fastkernels_norm")
    and hasattr(torch.ops.fastkernels_norm, "rmsnorm")
    and hasattr(torch.ops.fastkernels_norm, "fused_add_rmsnorm")
)


def _cast_weight(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if weight.dtype != x.dtype or weight.device != x.device:
        return weight.to(device=x.device, dtype=x.dtype)
    return weight


def _native_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    orig_dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = y.to(orig_dtype)
    return y * weight


def _native_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    y = x.float() + residual.float()
    residual_out = y.to(orig_dtype)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = y.to(orig_dtype) * weight
    return y, residual_out


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

    def forward(self, positions, hidden_states, residual, rotary_emb):
        input_norm = self.input_layernorm
        input_weight = _cast_weight(input_norm.weight, hidden_states)

        if residual is None:
            residual = hidden_states
            if hidden_states.is_cuda and _HAS_VLLM_RMS:
                normed = torch.empty_like(hidden_states)
                torch.ops._C.rms_norm(normed, hidden_states, input_weight, input_norm.eps)
                hidden_states = normed
            elif hidden_states.is_cuda and _HAS_FK_RMS:
                normed = torch.empty_like(hidden_states)
                torch.ops.fastkernels_norm.rmsnorm(
                    normed, hidden_states, input_weight, input_norm.eps
                )
                hidden_states = normed
            else:
                hidden_states = _native_rms_norm(hidden_states, input_weight, input_norm.eps)
        else:
            if hidden_states.is_cuda and _HAS_VLLM_RMS:
                torch.ops._C.fused_add_rms_norm(
                    hidden_states, residual, input_weight, input_norm.eps
                )
            elif hidden_states.is_cuda and _HAS_FK_RMS:
                torch.ops.fastkernels_norm.fused_add_rmsnorm(
                    hidden_states, residual, input_weight, input_norm.eps
                )
            else:
                hidden_states, residual = _native_fused_add_rms_norm(
                    hidden_states, residual, input_weight, input_norm.eps
                )

        hidden_states = self.self_attn(positions, hidden_states, rotary_emb=rotary_emb)

        post_norm = self.post_attention_layernorm
        post_weight = _cast_weight(post_norm.weight, hidden_states)
        if hidden_states.is_cuda and _HAS_VLLM_RMS:
            torch.ops._C.fused_add_rms_norm(
                hidden_states, residual, post_weight, post_norm.eps
            )
        elif hidden_states.is_cuda and _HAS_FK_RMS:
            torch.ops.fastkernels_norm.fused_add_rmsnorm(
                hidden_states, residual, post_weight, post_norm.eps
            )
        else:
            hidden_states, residual = _native_fused_add_rms_norm(
                hidden_states, residual, post_weight, post_norm.eps
            )

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
