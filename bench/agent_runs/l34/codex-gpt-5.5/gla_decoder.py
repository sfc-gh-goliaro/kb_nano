from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.gla_attention import GatedLinearAttention
from fastkernels.tasks.baseline.L2.gla_mlp import GLAMLP

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=0)
        inv = tl.rsqrt(ss / n_cols + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        y = x * inv * w
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


    @triton.jit
    def _add_rms_norm_kernel(
        a_ptr,
        b_ptr,
        w_ptr,
        out_ptr,
        y_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        idx = row * n_cols + offs
        a = tl.load(a_ptr + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
        v = (a + b).to(tl.float32)
        ss = tl.sum(v * v, axis=0)
        inv = tl.rsqrt(ss / n_cols + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + idx, v, mask=mask)
        tl.store(y_ptr + idx, v * inv * w, mask=mask)


def _num_warps(block: int) -> int:
    if block >= 4096:
        return 8
    if block >= 2048:
        return 4
    return 1


def _can_use_triton(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and weight.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.ndim >= 2
        and x.shape[-1] == weight.numel()
        and x.shape[-1] <= 65536
        and x.numel() > 0
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _fallback_rms_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return norm(x.reshape(-1, x.size(-1))).reshape_as(x)


def _fast_rms_norm(norm: nn.Module, x: torch.Tensor, eps: float) -> torch.Tensor:
    weight = norm.weight
    if not _can_use_triton(x, weight):
        return _fallback_rms_norm(norm, x)
    n_cols = x.shape[-1]
    rows = x.numel() // n_cols
    y = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    _rms_norm_kernel[(rows,)](x, weight, y, n_cols, eps, BLOCK=block, num_warps=_num_warps(block))
    return y


def _fast_add_rms_norm(
    norm: nn.Module,
    residual: torch.Tensor,
    update: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = norm.weight
    if not (
        _can_use_triton(residual, weight)
        and update.is_cuda
        and update.is_contiguous()
        and update.shape == residual.shape
        and update.dtype == residual.dtype
    ):
        out = residual + update
        return out, _fallback_rms_norm(norm, out)
    n_cols = residual.shape[-1]
    rows = residual.numel() // n_cols
    out = torch.empty_like(residual)
    y = torch.empty_like(residual)
    block = triton.next_power_of_2(n_cols)
    _add_rms_norm_kernel[(rows,)](
        residual,
        update,
        weight,
        out,
        y,
        n_cols,
        eps,
        BLOCK=block,
        num_warps=_num_warps(block),
    )
    return out, y


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
        self._norm_eps = float(config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        residual = hidden_states
        h = _fast_rms_norm(self.attn_norm, hidden_states, self._norm_eps)
        h, attentions, past_key_values = self.attn(
            hidden_states=h,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states, h = _fast_add_rms_norm(self.mlp_norm, residual, h, self._norm_eps)
        h = self.mlp(h)
        h.add_(hidden_states)
        return h, attentions, past_key_values
