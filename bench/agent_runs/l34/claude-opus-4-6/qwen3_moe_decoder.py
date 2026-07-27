from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.qwen3_moe import Qwen3MoE


@triton.jit
def _rms_norm_fwd(
    X, OUT, W,
    stride,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * stride + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

    rvar = tl.sum(x * x, axis=0) * (1.0 / N)
    rrms = tl.rsqrt(rvar + eps)
    out = (x * rrms) * w

    tl.store(OUT + off, out, mask=mask)


@triton.jit
def _fused_add_rms_norm_fwd(
    X, R, W,
    stride,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * stride + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

    s = x + r
    tl.store(R + off, s, mask=mask)

    rvar = tl.sum(s * s, axis=0) * (1.0 / N)
    rrms = tl.rsqrt(rvar + eps)
    out = (s * rrms) * w

    tl.store(X + off, out, mask=mask)


class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp = Qwen3MoE(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._N = config.hidden_size
        self._eps = config.rms_norm_eps
        bn = triton.next_power_of_2(config.hidden_size)
        self._BN = bn
        self._NW = 4 if bn <= 1024 else (8 if bn <= 4096 else 16)

    def forward(self, positions, hidden_states, residual):
        N = self._N
        BN = self._BN
        NW = self._NW
        eps = self._eps
        rows = hidden_states.shape[0]
        stride = hidden_states.stride(0)
        in_w = self.input_layernorm.weight

        if residual is None:
            residual = hidden_states
            hidden_states = torch.empty_like(hidden_states)
            _rms_norm_fwd[(rows,)](
                residual, hidden_states, in_w,
                stride, eps, N,
                BLOCK_N=BN, num_warps=NW,
            )
        else:
            _fused_add_rms_norm_fwd[(rows,)](
                hidden_states, residual, in_w,
                stride, eps, N,
                BLOCK_N=BN, num_warps=NW,
            )

        hidden_states = self.self_attn(positions, hidden_states)

        stride = hidden_states.stride(0)
        rows = hidden_states.shape[0]
        _fused_add_rms_norm_fwd[(rows,)](
            hidden_states, residual,
            self.post_attention_layernorm.weight,
            stride, eps, N,
            BLOCK_N=BN, num_warps=NW,
        )

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
