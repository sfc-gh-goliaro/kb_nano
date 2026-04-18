"""Kimi-Linear / Qwen3-Next hybrid state cache.

Mirrors vLLM's hybrid-model layout: KDA / GDN linear-attention layers use
**flat slotted state tensors** (one set of tensors per layer, indexed by a
per-sequence ``state_slot``); MLA full-attention layers use **paged KV
cache** (block-table per sequence indexes into per-layer ``[num_blocks,
block_size, num_heads, head_dim]`` tensors).

This replaces the old ``list[dict]`` per-slot layout that forced the
engine to invoke the model once per sequence. Now a single batched
forward with ``cu_seqlens`` + ``state_indices`` can run the entire
batch through both KDA and MLA layers.

State allocation per layer:

  KDA (Kimi-Linear):
    conv_q[i]: [num_slots, local_proj_dim, W]   bf16   (W = conv kernel size)
    conv_k[i]: [num_slots, local_proj_dim, W]   bf16
    conv_v[i]: [num_slots, local_proj_dim, W]   bf16
    recurrent[i]: [num_slots, local_num_heads, head_dim, head_dim]  fp32

  MLA (Kimi-Linear):
    k_cache[i]: [num_blocks, block_size, local_num_heads, qk_head_dim]  bf16
    v_cache[i]: [num_blocks, block_size, local_num_heads, qk_head_dim]  bf16
                  (V is zero-padded to qk_head_dim so K and V share head
                  dim and we can use the standard FA3 varlen kernel.)

  GDN (Qwen3-Next):
    conv[i]:    [num_slots, local_conv_dim, W]   bf16
    recurrent[i]: [num_slots, local_v_heads, head_v_dim, head_k_dim]  fp32 by default

The ``conv_*`` layout matches fla's ``causal_conv1d`` initial / final
state shape (``[N, D, W]``) so per-sequence gather/scatter is a single
``index_select`` / ``index_copy_`` on dim 0.
"""

from __future__ import annotations

from collections import deque

import torch


# Sentinel slot id written into ``slot_mapping`` for tokens that should not
# be persisted into the paged KV cache. Mirrors vLLM's PAD_SLOT_ID.
PAD_SLOT_ID = -1


class KimiLinearStateManager:
    """Flat slotted state + paged KV cache for Kimi-Linear / Qwen3-Next."""

    def __init__(
        self,
        *,
        config,
        num_slots: int,
        block_size: int,
        num_mla_blocks: int,
        tp_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_slots = num_slots
        self.block_size = block_size
        self.num_mla_blocks = num_mla_blocks
        self.tp_size = tp_size
        self.device = device
        self.dtype = dtype
        self.model_type = getattr(config, "model_type", "")

        self._free_slots: deque[int] = deque(range(num_slots))
        self._in_use_slots: set[int] = set()

        # KDA per-layer state (one entry per layer; ``None`` for non-KDA layers).
        self.conv_q: list[torch.Tensor | None] = [None] * self.num_layers
        self.conv_k: list[torch.Tensor | None] = [None] * self.num_layers
        self.conv_v: list[torch.Tensor | None] = [None] * self.num_layers
        # GDN per-layer state (Qwen3-Next packs Q/K/V into a single conv).
        self.gdn_conv: list[torch.Tensor | None] = [None] * self.num_layers
        # Recurrent (Delta-Net / GDN) state, per linear-attn layer.
        self.recurrent: list[torch.Tensor | None] = [None] * self.num_layers

        # MLA paged KV (one pair per MLA layer; ``None`` for KDA / linear layers).
        self.k_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.v_cache: list[torch.Tensor | None] = [None] * self.num_layers

        # MLA block pool. ``None`` for models with no full-attention layers.
        self._free_blocks: deque[int] | None = None
        if num_mla_blocks > 0:
            self._free_blocks = deque(range(num_mla_blocks))

        if self.model_type == "kimi_linear":
            self._allocate_kimi_linear()
        elif self.model_type == "qwen3_next":
            self._allocate_qwen3_next()
        else:
            raise ValueError(f"Unsupported hybrid model_type: {self.model_type!r}")

        # Track which layers belong to which family so callers can iterate.
        self.kda_layer_ids: list[int] = [
            i for i in range(self.num_layers) if self.recurrent[i] is not None
            and self.conv_q[i] is not None
        ]
        self.gdn_layer_ids: list[int] = [
            i for i in range(self.num_layers) if self.gdn_conv[i] is not None
        ]
        self.mla_layer_ids: list[int] = [
            i for i in range(self.num_layers) if self.k_cache[i] is not None
        ]

    def _allocate_kimi_linear(self) -> None:
        cfg = self.config
        local_kda_heads = cfg.kda_num_heads // self.tp_size
        local_kda_proj = cfg.kda_num_heads * cfg.kda_head_dim // self.tp_size
        K = cfg.short_conv_kernel_size

        local_mla_heads = cfg.num_attention_heads // self.tp_size
        qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        v_head_dim = cfg.v_head_dim

        for i in range(self.num_layers):
            if cfg.is_kda_layer(i):
                # fla causal_conv1d initial / final state layout: [N, D, W],
                # where W is the conv kernel size (e.g. 4) and D is the local
                # projection dim. We allocate one slot per active sequence.
                self.conv_q[i] = torch.zeros(
                    self.num_slots, local_kda_proj, K,
                    device=self.device, dtype=self.dtype,
                )
                self.conv_k[i] = torch.zeros(
                    self.num_slots, local_kda_proj, K,
                    device=self.device, dtype=self.dtype,
                )
                self.conv_v[i] = torch.zeros(
                    self.num_slots, local_kda_proj, K,
                    device=self.device, dtype=self.dtype,
                )
                self.recurrent[i] = torch.zeros(
                    self.num_slots, local_kda_heads,
                    cfg.kda_head_dim, cfg.kda_head_dim,
                    device=self.device, dtype=torch.float32,
                )
            else:
                # MLA layer.
                # NOTE: V is padded to qk_head_dim so that K and V share
                # the same head dim and we can use vLLM's standard FA3
                # ``flash_attn_varlen_func`` (which requires symmetric
                # head dims). The MLA layer slices ``[:v_head_dim]`` off
                # the per-token output before ``o_proj``. This matches
                # vLLM's "naive" (non-W_UK-absorbed) MLA path.
                self.k_cache[i] = torch.zeros(
                    self.num_mla_blocks, self.block_size,
                    local_mla_heads, qk_head_dim,
                    device=self.device, dtype=self.dtype,
                )
                self.v_cache[i] = torch.zeros(
                    self.num_mla_blocks, self.block_size,
                    local_mla_heads, qk_head_dim,
                    device=self.device, dtype=self.dtype,
                )

    def _allocate_qwen3_next(self) -> None:
        cfg = self.config
        local_k_heads = cfg.linear_num_key_heads // self.tp_size
        local_v_heads = cfg.linear_num_value_heads // self.tp_size
        head_k_dim = cfg.linear_key_head_dim
        head_v_dim = cfg.linear_value_head_dim
        K = cfg.linear_conv_kernel_dim
        # Qwen3-Next conv is on concat([q,k,v]) where v has v_per_k as many heads.
        conv_dim = (
            (local_k_heads * head_k_dim) * 2  # q, k
            + (local_v_heads * head_v_dim)
        )

        local_attn_heads = cfg.num_attention_heads // self.tp_size
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        local_kv_heads = max(1, cfg.num_key_value_heads // self.tp_size)

        for i in range(self.num_layers):
            if cfg.is_linear_attn_layer(i):
                # See KDA above: fla conv state is [N, D, W].
                self.gdn_conv[i] = torch.zeros(
                    self.num_slots, conv_dim, K,
                    device=self.device, dtype=self.dtype,
                )
                self.recurrent[i] = torch.zeros(
                    self.num_slots, local_v_heads, head_v_dim, head_k_dim,
                    device=self.device, dtype=torch.bfloat16,
                )
            else:
                # Standard MHA layer: K and V share the same head_dim.
                self.k_cache[i] = torch.zeros(
                    self.num_mla_blocks, self.block_size,
                    local_kv_heads, head_dim,
                    device=self.device, dtype=self.dtype,
                )
                self.v_cache[i] = torch.zeros(
                    self.num_mla_blocks, self.block_size,
                    local_kv_heads, head_dim,
                    device=self.device, dtype=self.dtype,
                )

    # ----- slot lifecycle -------------------------------------------------

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def can_allocate(self, num_tokens: int) -> bool:
        """Can we fit a sequence of ``num_tokens`` (fresh allocation)?"""
        if not self._free_slots:
            return False
        if self._free_blocks is None:
            return True
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return len(self._free_blocks) >= blocks_needed

    def can_extend(self, seq, num_new_tokens: int) -> bool:
        """Can we extend an already-allocated sequence by ``num_new_tokens``?"""
        if self._free_blocks is None:
            return True
        total_after = seq.num_computed_tokens + num_new_tokens
        blocks_after = (total_after + self.block_size - 1) // self.block_size
        new_blocks_needed = max(0, blocks_after - len(seq.block_table))
        return len(self._free_blocks) >= new_blocks_needed

    def allocate(self, seq) -> int:
        if getattr(seq, "state_slot", None) is not None:
            return seq.state_slot
        if not self._free_slots:
            raise RuntimeError("No free Kimi-Linear state slots")
        slot = self._free_slots.popleft()
        self._in_use_slots.add(slot)
        self._reset_slot(slot)
        seq.state_slot = slot
        seq.block_table = []
        return slot

    def ensure_blocks_for(self, seq, total_tokens: int) -> None:
        """Allocate enough blocks so ``seq`` can hold ``total_tokens`` MLA KV."""
        if self._free_blocks is None:
            return
        blocks_needed = (total_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) < blocks_needed:
            if not self._free_blocks:
                raise RuntimeError(
                    "No free MLA KV cache blocks (need to add preemption)"
                )
            seq.block_table.append(self._free_blocks.popleft())

    def deallocate(self, seq) -> None:
        slot = getattr(seq, "state_slot", None)
        if slot is not None and slot in self._in_use_slots:
            self._in_use_slots.remove(slot)
            self._reset_slot(slot)
            self._free_slots.append(slot)
            seq.state_slot = None
        if self._free_blocks is not None and seq.block_table:
            self._free_blocks.extend(seq.block_table)
            seq.block_table = []

    def reset(self) -> None:
        """Return all slots and blocks to the free pool. Used between bench
        scenarios so prior sequence state can't leak into the next run."""
        for slot in range(self.num_slots):
            self._reset_slot(slot)
        self._in_use_slots.clear()
        self._free_slots = deque(range(self.num_slots))
        if self._free_blocks is not None:
            self._free_blocks = deque(range(self.num_mla_blocks))

    def _reset_slot(self, slot: int) -> None:
        for layer_id in self.kda_layer_ids:
            self.conv_q[layer_id][slot].zero_()
            self.conv_k[layer_id][slot].zero_()
            self.conv_v[layer_id][slot].zero_()
            self.recurrent[layer_id][slot].zero_()
        for layer_id in self.gdn_layer_ids:
            self.gdn_conv[layer_id][slot].zero_()
            self.recurrent[layer_id][slot].zero_()
        # MLA paged blocks are reused via block_table churn; no per-slot
        # zeroing needed because freed blocks are simply rebound to other
        # sequences and overwritten as those sequences write their KV.

    # ----- diagnostics ----------------------------------------------------

    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def num_free_blocks(self) -> int:
        return len(self._free_blocks) if self._free_blocks is not None else 0
