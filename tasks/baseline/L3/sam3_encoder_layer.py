"""Fusion encoder layer for SAM3.

Self-attention over flattened image tokens followed by cross-attention to
text/prompt features, then a feed-forward network. Pre-norm architecture.

Reference: sam3/model/encoder.py TransformerEncoderLayer (pre_norm path)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.gelu import GELU
from ..L2.sam3_cross_attention import Sam3CrossAttention


class Sam3EncoderLayer(nn.Module):
    """Single fusion encoder layer for SAM3.

    Pre-norm: LN -> self-attn -> residual -> LN -> cross-attn -> residual ->
    LN -> FFN -> residual.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = Sam3CrossAttention(d_model, n_head)
        self.cross_attn = Sam3CrossAttention(d_model, n_head)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.ffn = nn.Sequential(
            Linear(d_model, dim_feedforward, bias=True),
            GELU(),
            nn.Dropout(dropout),
            Linear(dim_feedforward, d_model, bias=True),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tgt: (B, L, D) image tokens.
            memory: (B, S, D) text/prompt tokens.
            tgt_key_padding_mask: (B, L) padding mask for tgt.
            memory_key_padding_mask: (B, S) padding mask for memory.
            query_pos: (B, L, D) positional encoding for tgt.
            pos: (B, S, D) positional encoding for memory (unused in base).

        Returns:
            (B, L, D) processed tokens.
        """
        # Self-attention
        x = self.norm1(tgt)
        q = k = x + query_pos if query_pos is not None else x
        x = self.self_attn(q, k, x, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(x)

        # Cross-attention to prompt/text
        x = self.norm2(tgt)
        x = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(x)

        # FFN
        x = self.norm3(tgt)
        x = self.ffn(x)
        tgt = tgt + self.dropout3(x)

        return tgt
