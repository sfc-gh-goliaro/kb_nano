"""Attention blocks for Oasis DiT and VAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.oasis_rotary import RotaryEmbedding, apply_rotary_emb
from .oasis_mlp import OasisMLP


class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        *,
        is_causal: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b t h w (n d) -> (b h w) n t d", n=self.heads)
        k = rearrange(k, "b t h w (n d) -> (b h w) n t d", n=self.heads)
        v = rearrange(v, "b t h w (n d) -> (b h w) n t d", n=self.heads)
        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        out = rearrange(out, "(b h w) n t d -> b t h w (n d)", b=bsz, h=height, w=width)
        return self.to_out(out.to(q.dtype))


class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b t h w (n d) -> (b t) n h w d", n=self.heads)
        k = rearrange(k, "b t h w (n d) -> (b t) n h w d", n=self.heads)
        v = rearrange(v, "b t h w (n d) -> (b t) n h w d", n=self.heads)
        freqs = self.rotary_emb.get_axial_freqs(height, width)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        q = rearrange(q, "(b t) n h w d -> (b t) n (h w) d", b=bsz, t=time)
        k = rearrange(k, "(b t) n h w d -> (b t) n (h w) d", b=bsz, t=time)
        v = rearrange(v, "(b t) n h w d -> (b t) n (h w) d", b=bsz, t=time)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = rearrange(out, "(b t) n (h w) d -> b t h w (n d)", b=bsz, h=height, w=width)
        return self.to_out(out.to(q.dtype))


class OasisVAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = RotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        self.register_buffer(
            "rotary_freqs",
            self.rotary.get_axial_freqs(frame_height, frame_width),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b (h w) (n d) -> b n h w d", h=self.frame_height, w=self.frame_width, n=self.num_heads)
        k = rearrange(k, "b (h w) (n d) -> b n h w d", h=self.frame_height, w=self.frame_width, n=self.num_heads)
        v = rearrange(v, "b (h w) (n d) -> b n h w d", h=self.frame_height, w=self.frame_width, n=self.num_heads)
        q = apply_rotary_emb(self.rotary_freqs, q)
        k = apply_rotary_emb(self.rotary_freqs, k)
        q = rearrange(q, "b n h w d -> b n (h w) d")
        k = rearrange(k, "b n h w d -> b n (h w) d")
        v = rearrange(v, "b n h w d -> b n (h w) d")
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b n s d -> b s (n d)")
        return self.proj(out)


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
        self.norm1 = LayerNorm(dim)
        self.attn = OasisVAEAttention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
