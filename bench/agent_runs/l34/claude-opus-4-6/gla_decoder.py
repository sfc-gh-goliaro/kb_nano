from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.gla_attention import GatedLinearAttention
from fastkernels.tasks.baseline.L2.gla_mlp import GLAMLP


@triton.jit
def _rms_norm_fwd(
    X, W, Y,
    stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    off = row * stride
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + off + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(Y + off + cols, (x * rrms * w).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rms_norm_fwd(
    Res, X, W, Out, Normed,
    stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    off = row * stride
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    r = tl.load(Res + off + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + off + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    s = r + x
    tl.store(Out + off + cols, s.to(Out.dtype.element_ty), mask=mask)
    rrms = tl.rsqrt(tl.sum(s * s, axis=0) / N + eps)
    tl.store(Normed + off + cols, (s * rrms * w).to(Normed.dtype.element_ty), mask=mask)


def _triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    M, N = x_flat.shape
    y = torch.empty_like(x_flat)
    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_fwd[(M,)](
        x_flat, weight, y,
        x_flat.stride(0), N, eps, BLOCK_N,
        num_warps=max(1, min(32, BLOCK_N // 256)),
    )
    return y.reshape(shape)


def _triton_fused_add_rms_norm(
    residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = residual.shape
    N = shape[-1]
    res_flat = residual.reshape(-1, N)
    x_flat = x.reshape(-1, N)
    M = res_flat.shape[0]
    out = torch.empty_like(res_flat)
    normed = torch.empty_like(res_flat)
    BLOCK_N = triton.next_power_of_2(N)
    _fused_add_rms_norm_fwd[(M,)](
        res_flat, x_flat, weight, out, normed,
        res_flat.stride(0), N, eps, BLOCK_N,
        num_warps=max(1, min(32, BLOCK_N // 256)),
    )
    return out.reshape(shape), normed.reshape(shape)


class GLADecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedLinearAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            decay_mode=getattr(config, "decay_mode", "learned_low_rank"),
            gate_low_rank_dim=getattr(config, "gate_low_rank_dim", 16),
            gate_logit_normalizer=getattr(config, "gate_logit_normalizer", 16),
            use_rotary=getattr(config, "use_rotary", False),
            rotary_base=getattr(config, "rotary_base", 10000.0),
            rotary_max_position=getattr(config, "max_position_embeddings", 8192),
            norm_eps=config.norm_eps,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(config.hidden_size, config.intermediate_size)
        self._norm_eps = config.norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        residual = hidden_states
        h = _triton_rms_norm(hidden_states, self.attn_norm.weight, self._norm_eps)
        h, attentions, past_key_values = self.attn(
            hidden_states=h,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states, h = _triton_fused_add_rms_norm(
            residual, h, self.mlp_norm.weight, self._norm_eps
        )
        hidden_states = hidden_states + self.mlp(h)
        return hidden_states, attentions, past_key_values
