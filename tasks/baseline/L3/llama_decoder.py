"""Decoder layer: attention + MLP with RMSNorm residual connections.

Unified across Llama, Qwen2, and Qwen3 architectures:
  - bias:    Qwen2 uses bias=True on QKV projection.
  - qk_norm: Qwen3 applies per-head RMSNorm to Q and K before RoPE.

Optional fused RMSNorm+FP8 path (KB_NANO_FUSED_NORM=1): fuses norm and
per-token-group FP8 quantization into a single Triton kernel. Saves one
kernel launch per norm but is counterproductive under CUDA graph mode
where launch overhead is already amortized. Beneficial in eager mode.
"""

from __future__ import annotations

import os

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.llama_mlp import LlamaMLP


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 bias: bool = False, qk_norm: bool = False):
        super().__init__()
        fp8 = getattr(config, "fp8_block_size", None)
        self._use_fused_norm_fp8 = False
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            bias=bias, qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            fp8_block_size=fp8,
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if fp8 is not None and os.environ.get("KB_NANO_FUSED_NORM") == "1":
            from ..L1.fused_rmsnorm_fp8 import FusedRMSNormFP8Quant
            self._fused_input_norm = FusedRMSNormFP8Quant(
                config.hidden_size, eps=config.rms_norm_eps,
                group_size=fp8[1],
            )
            self._fused_post_norm = FusedRMSNormFP8Quant(
                config.hidden_size, eps=config.rms_norm_eps,
                group_size=fp8[1],
            )
            # Share weights with the existing RMSNorm modules so weight
            # loading works without changes
            del self._fused_input_norm.weight
            del self._fused_post_norm.weight
            self._fused_input_norm.weight = self.input_layernorm.weight
            self._fused_post_norm.weight = self.post_attention_layernorm.weight
            self._use_fused_norm_fp8 = True

    def forward(self, positions, hidden_states, residual):
        if self._use_fused_norm_fp8:
            return self._forward_fused_fp8(positions, hidden_states, residual)

        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def _forward_fused_fp8(self, positions, hidden_states, residual):
        """FP8 path: fused RMSNorm + FP8 quant → skip internal quant in linear."""
        fp8_out, fp8_scale, residual = self._fused_input_norm(
            hidden_states, residual)

        # Attention with pre-quantized input
        hidden_states = self.self_attn.forward_fp8(
            positions, fp8_out, fp8_scale)

        # Post-attention layernorm + FP8 quant
        fp8_out, fp8_scale, residual = self._fused_post_norm(
            hidden_states, residual)

        # MLP with pre-quantized input
        hidden_states = self.mlp.forward_fp8(fp8_out, fp8_scale)
        return hidden_states, residual
