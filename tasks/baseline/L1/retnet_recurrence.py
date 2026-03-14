"""Naive recurrent retention kernel.

Pure PyTorch reference implementation. The per-head decay is fixed
(data-independent): γ_h = 1 - 2^(-5-h). State is maintained in float32
for numerical stability, outputs are cast back to input dtype.
"""

from __future__ import annotations

import torch


def naive_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive loop-based retention recurrence.

    Args:
        q:  [B, H, T, K]
        k:  [B, H, T, K]
        v:  [B, H, T, V]
        scale: query scaling factor (default: K**-0.5)
        initial_state: [B, H, K, V] initial recurrent state
        output_final_state: whether to return final state

    Returns:
        o: [B, H, T, V]  output
        final_state: [B, H, K, V] or None
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    # Per-head decay: γ_h = 1 - 2^(-5-h), shape [H]
    gamma = 1.0 - q.new_tensor(2.0, dtype=torch.float32).pow(
        -5.0 - q.new_tensor(range(H), dtype=torch.float32)
    )

    h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h = h + initial_state.float()

    for i in range(T):
        q_i = q[:, :, i] * scale               # [B, H, K]
        k_i = k[:, :, i]                        # [B, H, K]
        v_i = v[:, :, i]                        # [B, H, V]

        h = h * gamma[None, :, None, None] + k_i.float()[..., None] * v_i.float()[..., None, :]
        o[:, :, i] = (q_i.float()[..., None] * h).sum(-2).to(v.dtype)

    final_state = h if output_final_state else None
    return o, final_state
