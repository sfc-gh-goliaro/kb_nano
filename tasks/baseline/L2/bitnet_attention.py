"""BitNet attention block.

GQA attention with W1.58A8 BitLinear projections, NeoX-style RoPE, and an
RMSNorm sub-norm applied between the attention output and ``o_proj``.

Architecture (per Microsoft's ``microsoft/bitnet-b1.58-2B-4T``):

    x ─► BitLinear(q,k,v) ─► RoPE ─► Attention(KV cache) ─► RMSNorm ─► BitLinear(o)

Weight names match the HuggingFace checkpoint convention so that the shared
weight loader can populate them directly:

    self_attn.q_proj.weight        / .weight_scale
    self_attn.k_proj.weight        / .weight_scale
    self_attn.v_proj.weight        / .weight_scale
    self_attn.o_proj.weight        / .weight_scale
    self_attn.attn_sub_norm.weight
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.bitnet_linear import BitLinear
from ..L1.rms_norm import RMSNorm
from .attention_impl import Attention


class BitNetAttention(nn.Module):
    """W1.58A8 GQA attention with sub-norm and NeoX RoPE.

    No tensor parallelism: BitNet b1.58-2B has only 5 KV heads, which makes
    head-level TP awkward.  Weights are replicated when ``tp_size > 1``.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 rotary_emb: nn.Module,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb

        q_size = num_attention_heads * head_dim
        kv_size = num_key_value_heads * head_dim

        self.q_proj = BitLinear(hidden_size, q_size, bias=False)
        self.k_proj = BitLinear(hidden_size, kv_size, bias=False)
        self.v_proj = BitLinear(hidden_size, kv_size, bias=False)
        self.o_proj = BitLinear(q_size, hidden_size, bias=False)

        self.attn_sub_norm = RMSNorm(q_size, eps=rms_norm_eps)

        self.attn = Attention(
            self.num_heads, head_dim, head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        attn_output = self.attn_sub_norm(attn_output)
        return self.o_proj(attn_output)
