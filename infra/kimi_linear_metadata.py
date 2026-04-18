"""Per-step batched-forward metadata for hybrid Kimi-Linear / Qwen3-Next models.

Mirrors vLLM's ``GDNAttentionMetadata`` (KDA / GDN linear-attention path)
combined with the standard paged-KV metadata (MLA / MHA path), so a single
forward pass can run a mixed batch through every layer of the hybrid model
without any per-sequence Python loops.

Layout convention (matches vLLM):
  * Tokens are flat ``[num_actual_tokens, hidden_size]``; sequence boundaries
    are recoverable from ``query_start_loc``.
  * Within the batch, **decode requests come first** then **prefill requests**
    -- this matches vLLM's ``reorder_batch_to_split_decodes_and_prefills`` and
    lets KDA layers slice ``query_start_loc[: num_decodes + 1]`` for the
    decode-only fused-recurrent kernel.
  * ``state_indices`` is a 1-D int32 tensor of shape ``[batch_size]`` giving
    the per-sequence slot id used to index ``KimiLinearStateManager``'s flat
    ``conv_*`` / ``recurrent`` tensors.
  * ``slot_mapping`` is a 1-D int32 tensor of shape ``[num_actual_tokens]``
    with the *physical* paged-KV slot id (``block_id * block_size + offset``)
    where each token's MLA K/V should be written.
  * ``has_initial_state`` is only meaningful for prefill requests and is
    indexed in batch order (decodes first, then prefills) with length
    ``batch_size``; the linear-attention layer slices the prefill suffix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KimiLinearMetadata:
    """One-batch metadata for the hybrid forward pass."""

    # ----- common -------------------------------------------------------
    num_actual_tokens: int = 0
    batch_size: int = 0

    # Per-token positions, [num_actual_tokens] int64 — passed to RoPE / MLA.
    positions: torch.Tensor | None = None

    # Per-sequence prefix sum of query lengths, [batch_size + 1] int32.
    # query_start_loc[i+1] - query_start_loc[i] is the # tokens for seq i.
    query_start_loc: torch.Tensor | None = None
    # Same as ``query_start_loc`` but kept on CPU for kernels that need it
    # (some FLA helpers use it for chunk metadata).
    query_start_loc_cpu: torch.Tensor | None = None
    max_query_len: int = 0

    # Per-sequence total context length AFTER this step (including new
    # tokens), [batch_size] int32.
    seq_lens: torch.Tensor | None = None
    max_seq_len: int = 0

    # Per-sequence slot id into KimiLinearStateManager, [batch_size] int32.
    state_indices: torch.Tensor | None = None

    # ----- prefill / decode split (decode first, then prefill) ----------
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0

    # Bool mask, [batch_size]. ``True`` for sequences whose prior context
    # length > 0 (i.e. resumable prefill or any decode). Linear-attention
    # layers slice ``has_initial_state[num_decodes:]`` for the prefill kernel.
    has_initial_state: torch.Tensor | None = None

    # ----- MLA / paged KV ----------------------------------------------
    # Physical KV slot per token, [num_actual_tokens] int32.
    slot_mapping: torch.Tensor | None = None
    # Block-table per request, [batch_size, max_blocks] int32.
    block_tables: torch.Tensor | None = None

    # ----- output gather ------------------------------------------------
    # Indices into the flat hidden_states from which to read the per-seq
    # logit (last token of each request), [batch_size] int64.
    logit_indices: torch.Tensor | None = None


# Module-level current-batch metadata, set by the engine before invoking
# the model and read by KDA / MLA layers via ``get_metadata()``.
_METADATA: KimiLinearMetadata | None = None


def get_metadata() -> KimiLinearMetadata | None:
    return _METADATA


def set_metadata(md: KimiLinearMetadata | None) -> None:
    global _METADATA
    _METADATA = md


def reset_metadata() -> None:
    global _METADATA
    _METADATA = None
