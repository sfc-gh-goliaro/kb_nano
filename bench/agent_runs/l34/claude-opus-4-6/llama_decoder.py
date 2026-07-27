from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP

try:
    import vllm._C  # noqa: F401
    _rmsnorm = torch.ops._C.rms_norm
    _fused_add_rmsnorm = torch.ops._C.fused_add_rms_norm
except (ImportError, AttributeError):
    _rmsnorm = torch.ops.fastkernels_norm.rmsnorm
    _fused_add_rmsnorm = torch.ops.fastkernels_norm.fused_add_rmsnorm


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 bias: bool = False, qk_norm: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            bias=bias, qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        w1 = self.input_layernorm.weight
        e1 = self.input_layernorm.eps
        if residual is None:
            residual = hidden_states
            hidden_states = torch.empty_like(hidden_states)
            _rmsnorm(hidden_states, residual, w1, e1)
        else:
            _fused_add_rmsnorm(hidden_states, residual, w1, e1)

        hidden_states = self.self_attn(positions, hidden_states)

        w2 = self.post_attention_layernorm.weight
        e2 = self.post_attention_layernorm.eps
        _fused_add_rmsnorm(hidden_states, residual, w2, e2)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
