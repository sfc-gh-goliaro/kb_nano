from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.qwen3_moe import Qwen3MoE


if triton is not None:
    @triton.jit
    def _rms_norm_kernel(x, w, y, n_rows: tl.constexpr, hidden: tl.constexpr,
                         eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < hidden
        vals = tl.load(x + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(vals * vals, axis=0) / hidden
        inv = tl.rsqrt(ss + eps)
        wt = tl.load(w + offs, mask=mask, other=0.0).to(tl.float32)
        out = vals * inv * wt
        tl.store(y + row * hidden + offs, out, mask=mask)


    @triton.jit
    def _add_rms_norm_kernel(x, residual, w, y, res_out,
                             n_rows: tl.constexpr, hidden: tl.constexpr,
                             eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < hidden
        base = row * hidden + offs
        xv = tl.load(x + base, mask=mask, other=0.0).to(tl.float32)
        rv = tl.load(residual + base, mask=mask, other=0.0).to(tl.float32)
        summed = xv + rv
        ss = tl.sum(summed * summed, axis=0) / hidden
        inv = tl.rsqrt(ss + eps)
        wt = tl.load(w + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(res_out + base, summed, mask=mask)
        tl.store(y + base, summed * inv * wt, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _num_warps(block: int) -> int:
    if block >= 4096:
        return 8
    if block >= 2048:
        return 4
    return 1


class _FastRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.eps = eps

    def _torch_norm(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (out * self.weight.float()).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if (triton is None or not x.is_cuda or x.requires_grad or
                (residual is not None and residual.requires_grad) or
                not self.weight.is_cuda):
            if residual is None:
                return self._torch_norm(x)
            res = x + residual
            return self._torch_norm(res), res

        hidden = x.shape[-1]
        if hidden != self.weight.numel() or hidden > 131072:
            if residual is None:
                return self._torch_norm(x)
            res = x + residual
            return self._torch_norm(res), res

        x_contig = x if x.is_contiguous() else x.contiguous()
        n_rows = x_contig.numel() // hidden
        block = _next_power_of_2(hidden)
        y = torch.empty_like(x_contig)

        if residual is None:
            _rms_norm_kernel[(n_rows,)](
                x_contig, self.weight, y, n_rows, hidden,
                float(self.variance_epsilon), BLOCK=block,
                num_warps=_num_warps(block),
            )
            return y.reshape_as(x)

        residual_contig = residual if residual.is_contiguous() else residual.contiguous()
        res_out = torch.empty_like(x_contig)
        _add_rms_norm_kernel[(n_rows,)](
            x_contig, residual_contig, self.weight, y, res_out,
            n_rows, hidden, float(self.variance_epsilon), BLOCK=block,
            num_warps=_num_warps(block),
        )
        return y.reshape_as(x), res_out.reshape_as(x)


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
        self.input_layernorm = _FastRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _FastRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
