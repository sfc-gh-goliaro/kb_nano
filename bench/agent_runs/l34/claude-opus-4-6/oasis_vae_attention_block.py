from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.oasis_mlp import OasisMLP
from fastkernels.tasks.baseline.L2.oasis_vae_attention import OasisVAEAttention

try:
    from flash_attn import flash_attn_func
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False


@triton.jit
def _ln_kernel(x_ptr, o_ptr, w_ptr, b_ptr, N, eps, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < N
    base = row * stride
    raw = tl.load(x_ptr + base + c, mask=m, other=0.0)
    x = raw.to(tl.float32)
    mu = tl.sum(x, axis=0) / N
    d = tl.where(m, x - mu, 0.0)
    hat = d / tl.sqrt(tl.sum(d * d, axis=0) / N + eps)
    w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + base + c, (hat * w + b).to(raw.dtype), mask=m)


@triton.jit
def _add_ln_kernel(r_ptr, a_ptr, s_ptr, n_ptr, w_ptr, b_ptr, N, eps, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < N
    base = row * stride
    r = tl.load(r_ptr + base + c, mask=m, other=0.0)
    a = tl.load(a_ptr + base + c, mask=m, other=0.0)
    s = r + a
    tl.store(s_ptr + base + c, s, mask=m)
    x = s.to(tl.float32)
    mu = tl.sum(x, axis=0) / N
    d = tl.where(m, x - mu, 0.0)
    hat = d / tl.sqrt(tl.sum(d * d, axis=0) / N + eps)
    w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=m, other=0.0).to(tl.float32)
    tl.store(n_ptr + base + c, (hat * w + b).to(r.dtype), mask=m)


def _run_ln(x, weight, bias, eps):
    sh = x.shape
    N = sh[-1]
    f = x.contiguous().view(-1, N)
    o = torch.empty_like(f)
    BK = triton.next_power_of_2(N)
    _ln_kernel[(f.shape[0],)](f, o, weight, bias, N, eps, N, BLOCK=BK,
                               num_warps=max(1, min(32, BK // 256)))
    return o.view(sh)


def _run_add_ln(res, add, weight, bias, eps):
    sh = res.shape
    N = sh[-1]
    rf = res.contiguous().view(-1, N)
    af = add.contiguous().view(-1, N)
    s = torch.empty_like(rf)
    n = torch.empty_like(rf)
    BK = triton.next_power_of_2(N)
    _add_ln_kernel[(rf.shape[0],)](rf, af, s, n, weight, bias, N, eps, N, BLOCK=BK,
                                    num_warps=max(1, min(32, BK // 256)))
    return s.view(sh), n.view(sh)


class OasisVAEAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = OasisVAEAttention(
            dim, num_heads, frame_height, frame_width, qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)
        self._dim = dim
        self._nh = num_heads
        self._hd = dim // num_heads
        self._H = frame_height
        self._W = frame_width
        freqs = self.attn.rotary_freqs
        self.register_buffer("_rc", freqs.cos().unsqueeze(-2), persistent=False)
        self.register_buffer("_rs", freqs.sin().unsqueeze(-2), persistent=False)
        self._rd = freqs.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H, W, nh, hd, D, rd = self._H, self._W, self._nh, self._hd, self._dim, self._rd
        S = H * W

        xn = _run_ln(x, self.norm1.weight, self.norm1.bias, 1e-6)

        qkv = F.linear(xn, self.attn.qkv.weight, self.attn.qkv.bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, H, W, nh, hd)
        k = k.reshape(B, H, W, nh, hd)

        rc, rs = self._rc, self._rs
        qr = q[..., :rd]
        qh = qr.reshape(*qr.shape[:-1], -1, 2)
        q = torch.cat((qr * rc + torch.stack((-qh[..., 1], qh[..., 0]), -1).flatten(-2) * rs, q[..., rd:]), -1)
        kr = k[..., :rd]
        kh = kr.reshape(*kr.shape[:-1], -1, 2)
        k = torch.cat((kr * rc + torch.stack((-kh[..., 1], kh[..., 0]), -1).flatten(-2) * rs, k[..., rd:]), -1)

        q = q.reshape(B, S, nh, hd)
        k = k.reshape(B, S, nh, hd)
        v = v.reshape(B, S, nh, hd)

        if _HAS_FLASH and x.dtype != torch.float32:
            o = flash_attn_func(q, k, v, causal=False)
            if isinstance(o, tuple):
                o = o[0]
        else:
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False,
            ).transpose(1, 2)

        o = F.linear(o.reshape(B, S, D), self.attn.proj.weight, self.attn.proj.bias)
        x, xn = _run_add_ln(x, o, self.norm2.weight, self.norm2.bias, 1e-6)

        h = F.gelu(F.linear(xn, self.mlp.fc1.weight, self.mlp.fc1.bias), approximate="none")
        h = F.linear(h, self.mlp.fc2.weight, self.mlp.fc2.bias)
        return x + h
