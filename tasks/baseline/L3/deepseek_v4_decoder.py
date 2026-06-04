"""DeepSeek V4 decoder layer with MHC, MLA attention, and MoE.

Each layer:
  1. hc_pre(residual) → pre_mix, post_mix, comb_mix, layer_input
  2. attn_norm(layer_input) → attention → attn_output
  3. hc_post(attn_output, residual, post_mix, comb_mix) → new_residual
  4. Repeat for FFN

Reference: vllm/model_executor/models/deepseek_v4.py:DeepseekV4DecoderLayer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.rms_norm import RMSNorm
from ..L2.deepseek_v4_attention import DeepSeekV4Attention
from ..L2.deepseek_v4_moe import DeepSeekV4MoE
from ..L2.llama_mlp import LlamaMLP


def _mhc_pre_pytorch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch implementation of mhc_pre.

    Fallback for when tilelang/DeepGEMM aren't available.
    Uses vllm's torch.ops.vllm.mhc_pre when available for exact parity.
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    outer_shape = residual.shape[:-2]
    x = residual.view(-1, hc_mult, hidden_size)
    num_tokens = x.shape[0]

    # RMS normalization + linear projection
    x_flat = x.view(num_tokens, hc_mult * hidden_size).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x_flat, fn) * rsqrt  # (num_tokens, hc_mult3)

    # Split mixes into pre, post, comb
    pre_raw = mixes[:, :hc_mult]
    post_raw = mixes[:, hc_mult:hc_mult * 2]
    comb_raw = mixes[:, hc_mult * 2:]

    # Pre mix: sigmoid + eps → weighted sum → layer_input
    pre_mix = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    layer_input = (pre_mix.unsqueeze(-1) * x.float()).sum(dim=1).to(residual.dtype)

    # Post mix: sigmoid * alpha
    post_mix = (
        torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc_mult:hc_mult * 2])
        * hc_post_mult_value
    )

    # Comb mix: softmax + sinkhorn normalization
    cm = comb_raw * hc_scale[2] + hc_base[hc_mult * 2:]
    cm = cm.view(num_tokens, hc_mult, hc_mult)
    cm = cm.softmax(dim=-1) + hc_sinkhorn_eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = cm.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def _mhc_post_pytorch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch implementation of mhc_post.

    out[i] = sum_j(comb[j,i] * residual[j]) + post[i] * x
    """
    # comb_res_mix: (..., hc_mult, hc_mult), residual: (..., hc_mult, hidden)
    # post_layer_mix: (..., hc_mult, 1), x: (..., hidden)
    out = torch.einsum('...ji,...jh->...ih', comb_res_mix, residual.float())
    out = out + post_layer_mix * x.unsqueeze(-2).float()
    return out.to(residual.dtype)


# Try vllm's fused ops, fall back to PyTorch
_USE_VLLM_MHC = False
try:
    _mhc_pre_op = torch.ops.vllm.mhc_pre
    _mhc_post_op = torch.ops.vllm.mhc_post
    _USE_VLLM_MHC = True
except (AttributeError, RuntimeError):
    pass


class DeepSeekV4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int,
                 rotary_emb: nn.Module,
                 quant_config: dict | None = None,
                 topk_indices_buffer: torch.Tensor | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_post_alpha = 2.0

        compress_ratio = 0
        if config.compress_ratios and layer_idx < len(config.compress_ratios):
            compress_ratio = config.compress_ratios[layer_idx]
        compress_ratio = max(compress_ratio, 1) if compress_ratio > 0 else 0

        self.attn = DeepSeekV4Attention(
            config, rotary_emb=rotary_emb,
            quant_config=quant_config,
            compress_ratio=compress_ratio,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.ffn = DeepSeekV4MoE(
            config, layer_idx=layer_idx, quant_config=quant_config,
        )

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MHC parameters
        hc_dim = config.hc_mult * config.hidden_size
        mix_hc = (2 + config.hc_mult) * config.hc_mult

        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        if _USE_VLLM_MHC:
            post_mix, comb_mix, layer_input = _mhc_pre_op(
                x, hc_fn, hc_scale, hc_base,
                self.rms_norm_eps, self.hc_eps, self.hc_eps,
                self.hc_post_alpha, self.hc_sinkhorn_iters,
            )
        else:
            post_mix, comb_mix, layer_input = _mhc_pre_pytorch(
                x, hc_fn, hc_scale, hc_base,
                self.rms_norm_eps, self.hc_eps, self.hc_eps,
                self.hc_post_alpha, self.hc_sinkhorn_iters,
            )
        return layer_input, post_mix, comb_mix

    def hc_post(self, x, residual, post, comb):
        if _USE_VLLM_MHC:
            return _mhc_post_op(x, residual, post, comb)
        return _mhc_post_pytorch(x, residual, post, comb)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                input_ids: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x
