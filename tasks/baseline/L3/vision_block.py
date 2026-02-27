"""Vision transformer block for Qwen VL models.

Uses LayerNorm (not RMSNorm) with pre-norm residual connections,
encoder-only attention, and vision MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.vision_attention import VisionAttention
from ..L2.vision_mlp import VisionMLP, Qwen3VisionMLP


class VisionBlock(nn.Module):
    """Qwen2-VL vision transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = VisionAttention(embed_dim, num_heads)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = VisionMLP(embed_dim, mlp_hidden)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens,
            rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VisionBlock(nn.Module):
    """Qwen3-VL vision transformer block with configurable MLP."""

    def __init__(self, embed_dim: int, num_heads: int,
                 intermediate_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = Qwen3VisionMLP(embed_dim, intermediate_size)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens,
            rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x
