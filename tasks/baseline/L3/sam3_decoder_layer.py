"""Detection decoder layer for SAM3.

Self-attention over object queries, cross-attention to encoder memory, optional
cross-attention to text, and feed-forward network. Pre-norm architecture with
iterative box refinement support.

Reference: sam3/model/decoder.py TransformerDecoderLayer (pre_norm path)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.gelu import GELU
from ..L2.sam3_cross_attention import Sam3CrossAttention


class Sam3DecoderLayer(nn.Module):
    """Single detection decoder layer for SAM3.

    Pre-norm: self-attn -> cross-attn (memory) -> cross-attn (text) -> FFN.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
        text_cross_attention: Whether to include text cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        text_cross_attention: bool = True,
    ):
        super().__init__()
        self.self_attn = Sam3CrossAttention(d_model, n_head)
        self.cross_attn_memory = Sam3CrossAttention(d_model, n_head)

        self.has_text_cross_attn = text_cross_attention
        if text_cross_attention:
            self.cross_attn_text = Sam3CrossAttention(d_model, n_head)
            self.norm_text = LayerNorm(d_model)
            self.dropout_text = nn.Dropout(dropout)

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
        memory_text: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tgt: (B, Q, D) object queries.
            memory: (B, L, D) encoder memory.
            tgt_key_padding_mask: (B, Q) padding mask for queries.
            memory_key_padding_mask: (B, L) padding mask for memory.
            memory_text: (B, S, D) text features for text cross-attention.
            text_attention_mask: (B, S) padding mask for text.

        Returns:
            (B, Q, D) updated object queries.
        """
        # Self-attention
        x = self.norm1(tgt)
        x = self.self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(x)

        # Cross-attention to encoder memory
        x = self.norm2(tgt)
        x = self.cross_attn_memory(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(x)

        # Cross-attention to text (optional)
        if self.has_text_cross_attn and memory_text is not None:
            x = self.norm_text(tgt)
            x = self.cross_attn_text(x, memory_text, memory_text, key_padding_mask=text_attention_mask)
            tgt = tgt + self.dropout_text(x)

        # FFN
        x = self.norm3(tgt)
        x = self.ffn(x)
        tgt = tgt + self.dropout3(x)

        return tgt
