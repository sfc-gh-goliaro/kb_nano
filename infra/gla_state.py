"""GLA state-slot cache management."""

from __future__ import annotations

from collections import deque

import torch


class GLASlotCache:
    """Cache view for one GLA sequence slot — list of per-layer recurrent states."""

    def __init__(self, states: list[torch.Tensor]):
        self.states = states

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        return self.states[layer_idx]

    def __setitem__(self, layer_idx: int, value: torch.Tensor):
        self.states[layer_idx].zero_()
        self.states[layer_idx] += value.to(self.states[layer_idx].device)


class GLAStateManager:
    """Owns global GLA recurrent state tensors and free slot bookkeeping."""

    def __init__(
        self,
        *,
        num_hidden_layers: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_slots = num_slots
        self.num_layers = num_hidden_layers
        self._free_slots: deque[int] = deque(range(num_slots))
        self._in_use: set[int] = set()
        self._slot_views: dict[int, GLASlotCache] = {}

        # Pre-allocate [num_slots, num_heads, head_k_dim, head_v_dim] per layer
        self.recurrent_states: list[torch.Tensor] = []
        for _ in range(num_hidden_layers):
            state = torch.zeros(
                num_slots, num_heads, head_k_dim, head_v_dim,
                device=device, dtype=dtype,
            )
            self.recurrent_states.append(state)

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def reset_slot(self, slot: int) -> None:
        for layer_idx in range(self.num_layers):
            self.recurrent_states[layer_idx][slot].zero_()

    def allocate(self, seq) -> int:
        if getattr(seq, "state_slot", None) is not None:
            return seq.state_slot
        if not self._free_slots:
            raise RuntimeError("No free GLA state slots")
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

    def get_slot_cache(self, slot: int) -> GLASlotCache:
        cache = self._slot_views.get(slot)
        if cache is not None:
            return cache
        views = [s[slot:slot + 1] for s in self.recurrent_states]
        cache = GLASlotCache(views)
        self._slot_views[slot] = cache
        return cache
