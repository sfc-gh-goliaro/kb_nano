"""Global inference context for paged KV cache coordination."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    # Chunked prefill: mixed batch with both prefill and decode tokens
    is_mixed: bool = False
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    # Decode-specific fields for mixed batches (indexed over decode tokens only)
    decode_context_lens: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables)


def set_mixed_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                      slot_mapping, num_prefill_tokens, num_decode_tokens,
                      decode_context_lens, decode_block_tables):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=True, is_mixed=True,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        decode_context_lens=decode_context_lens,
        decode_block_tables=decode_block_tables,
        block_tables=decode_block_tables,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
