from __future__ import annotations
from typing import Callable

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.quickgelu import QuickGELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.vision_attention import VisionAttention
from fastkernels.tasks.baseline.L2.vision_mlp import VisionMLP


@triton.jit
def _ln_fwd(X, Y, W, B, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * N
    x = tl.load(X + off + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    y = xc * tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + off + cols, y * w + b, mask=mask)


@triton.jit
def _add_ln_fwd(X, R, O, NO, W, B, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * N
    x = tl.load(X + off + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + off + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + r
    tl.store(O + off + cols, x, mask=mask)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    y = xc * tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(NO + off + cols, y * w + b, mask=mask)


def _nw(n):
    if n <= 512:
        return 2
    if n <= 1024:
        return 4
    if n <= 2048:
        return 8
    return 16


def _fast_ln(x, w, b, eps):
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N)
    y = torch.empty_like(x2)
    BN = triton.next_power_of_2(N)
    _ln_fwd[(x2.shape[0],)](x2, y, w, b, N, eps, BLOCK_N=BN, num_warps=_nw(BN))
    return y.reshape(shape)


def _fused_add_ln(x, res, w, b, eps):
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N)
    r2 = res.reshape(-1, N)
    o = torch.empty_like(x2)
    no = torch.empty_like(x2)
    BN = triton.next_power_of_2(N)
    _add_ln_fwd[(x2.shape[0],)](x2, r2, o, no, w, b, N, eps, BLOCK_N=BN, num_warps=_nw(BN))
    return o.reshape(shape), no.reshape(shape)


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_hidden_dim: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)
        self._eps = norm_eps

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        normed = _fast_ln(x, self.norm1.weight, self.norm1.bias, self._eps)
        attn_out = self.attn(normed, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen)
        x, normed = _fused_add_ln(x, attn_out, self.norm2.weight, self.norm2.bias, self._eps)
        x = x + self.mlp(normed)
        return x
