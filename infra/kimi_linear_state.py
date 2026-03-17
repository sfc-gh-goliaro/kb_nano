"""Kimi-Linear state-slot cache management.

Manages per-sequence state for hybrid KDA+MLA model:
  - KDA layers: conv states (conv_q/k/v) + recurrent state (Delta-Net h)
  - MLA layers: growing KV cache (k_cache, v_cache tensors)

State is stored as a list of dicts (one per layer). The model populates
dicts during forward; on deallocation we clear them to free GPU memory.
"""

from __future__ import annotations

from collections import deque


class KimiLinearStateManager:
    """Slot-based state manager for Kimi-Linear hybrid model."""

    def __init__(self, *, num_hidden_layers: int, num_slots: int):
        self.num_layers = num_hidden_layers
        self.num_slots = num_slots
        self._free_slots: deque[int] = deque(range(num_slots))
        self._in_use: set[int] = set()
        # Each slot is a list of dicts (one per layer)
        self._slot_states: list[list[dict]] = [
            [{} for _ in range(num_hidden_layers)] for _ in range(num_slots)
        ]

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def reset_slot(self, slot: int) -> None:
        for d in self._slot_states[slot]:
            d.clear()

    def allocate(self, seq) -> int:
        if getattr(seq, "state_slot", None) is not None:
            return seq.state_slot
        if not self._free_slots:
            raise RuntimeError("No free Kimi-Linear state slots")
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

    def get_slot_cache(self, slot: int) -> list[dict]:
        """Returns list of per-layer state dicts for this slot."""
        return self._slot_states[slot]
