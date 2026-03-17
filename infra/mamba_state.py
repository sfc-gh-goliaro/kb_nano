"""Runtime Mamba state-slot cache management (non-benchmark infrastructure)."""

from __future__ import annotations

from collections import deque

import torch


class MambaSlotCache:
    """Cache view for one sequence slot."""

    def __init__(
        self,
        conv_states: list[torch.Tensor],
        ssm_states: list[torch.Tensor],
        conv_kernel_size: int,
    ):
        self.conv_states = conv_states
        self.ssm_states = ssm_states
        self.conv_kernel_size = conv_kernel_size

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)
        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(
            device=conv_state.device,
            dtype=conv_state.dtype,
        )
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx].zero_()
        self.ssm_states[layer_idx] += new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]


class MambaStateManager:
    """Owns global Mamba recurrent state tensors and free slot bookkeeping."""

    def __init__(
        self,
        *,
        num_hidden_layers: int,
        conv_dim: int,
        ssm_state_shape: tuple[int, ...],
        conv_kernel: int,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_slots = num_slots
        self.conv_kernel = conv_kernel
        self._free_slots: deque[int] = deque(range(num_slots))
        self._in_use: set[int] = set()
        self._slot_views: dict[int, MambaSlotCache] = {}

        self.conv_states: list[torch.Tensor] = []
        self.ssm_states: list[torch.Tensor] = []
        for _ in range(num_hidden_layers):
            conv_state = torch.zeros(
                num_slots,
                conv_dim,
                conv_kernel,
                device=device,
                dtype=dtype,
            )
            ssm_state = torch.zeros(
                num_slots,
                *ssm_state_shape,
                device=device,
                dtype=dtype,
            )
            if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "mark_static_address"):
                torch._dynamo.mark_static_address(conv_state)
                torch._dynamo.mark_static_address(ssm_state)
            self.conv_states.append(conv_state)
            self.ssm_states.append(ssm_state)

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def reset_slot(self, slot: int) -> None:
        for layer_idx in range(len(self.conv_states)):
            self.conv_states[layer_idx][slot].zero_()
            self.ssm_states[layer_idx][slot].zero_()

    def allocate(self, seq) -> int:
        if getattr(seq, "state_slot", None) is not None:
            return seq.state_slot
        if not self._free_slots:
            raise RuntimeError("No free Mamba state slots")
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

    def get_slot_cache(self, slot: int) -> MambaSlotCache:
        cache = self._slot_views.get(slot)
        if cache is not None:
            return cache

        conv_views = [x[slot:slot + 1] for x in self.conv_states]
        ssm_views = [x[slot:slot + 1] for x in self.ssm_states]
        cache = MambaSlotCache(conv_views, ssm_views, self.conv_kernel)
        self._slot_views[slot] = cache
        return cache
