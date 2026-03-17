"""RWKV7 state-slot cache management."""

from __future__ import annotations

from collections import deque

import torch


class RWKV7SlotCache:
    """Cache view for one RWKV7 sequence slot."""

    def __init__(
        self,
        conv_states: list[torch.Tensor],
        recurrent_states: list[torch.Tensor],
    ):
        self.conv_states = conv_states
        self.recurrent_states = recurrent_states


class RWKV7StateManager:
    """Owns global RWKV7 recurrent + conv state tensors and free slot bookkeeping."""

    def __init__(
        self,
        *,
        num_hidden_layers: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        head_v_dims: list[int],
        num_slots: int,
        conv_dtype: torch.dtype,
        recurrent_dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_slots = num_slots
        self.num_layers = num_hidden_layers
        self._free_slots: deque[int] = deque(range(num_slots))
        self._in_use: set[int] = set()
        self._slot_views: dict[int, RWKV7SlotCache] = {}

        # Conv states: [num_slots, hidden_size, 2] per layer (shift state)
        self.conv_states: list[torch.Tensor] = []
        # Recurrent states: [num_slots, num_heads, head_dim, head_v_dim] per layer
        self.recurrent_states: list[torch.Tensor] = []
        for i in range(num_hidden_layers):
            conv_state = torch.zeros(
                num_slots, hidden_size, 2,
                device=device, dtype=conv_dtype,
            )
            rec_state = torch.zeros(
                num_slots, num_heads, head_dim, head_v_dims[i],
                device=device, dtype=recurrent_dtype,
            )
            self.conv_states.append(conv_state)
            self.recurrent_states.append(rec_state)

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def reset_slot(self, slot: int) -> None:
        for i in range(self.num_layers):
            self.conv_states[i][slot].zero_()
            self.recurrent_states[i][slot].zero_()

    def allocate(self, seq) -> int:
        if getattr(seq, "state_slot", None) is not None:
            return seq.state_slot
        if not self._free_slots:
            raise RuntimeError("No free RWKV7 state slots")
        slot = self._free_slots.popleft()
        self._in_use.add(slot)
        self.reset_slot(slot)
        seq.state_slot = slot
        return slot

    def deallocate(self, seq) -> None:
        slot = getattr(seq, "state_slot", None)
        if slot is None:
            return
        if slot in self._in_use:
            self._in_use.remove(slot)
            self.reset_slot(slot)
            self._free_slots.append(slot)
        seq.state_slot = None

    def get_slot_cache(self, slot: int) -> RWKV7SlotCache:
        cache = self._slot_views.get(slot)
        if cache is not None:
            return cache
        conv_views = [c[slot:slot + 1] for c in self.conv_states]
        rec_views = [r[slot:slot + 1] for r in self.recurrent_states]
        cache = RWKV7SlotCache(conv_views, rec_views)
        self._slot_views[slot] = cache
        return cache
