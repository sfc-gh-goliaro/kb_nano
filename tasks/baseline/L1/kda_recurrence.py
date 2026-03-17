"""Naive recurrent KDA (Kimi Delta Attention) kernel.

Pure PyTorch reference implementation of the Delta-Net recurrence.
State is maintained in float32 for numerical stability.
Q and K are L2-normalized inside the kernel, then Q is scaled by 1/sqrt(K).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive loop-based KDA (Delta-Net) recurrence.

    Implements: S = decay * S + beta * k ⊗ (v - k^T @ S)
                o = q^T @ S

    Args:
        q:  [B, H, T, K]
        k:  [B, H, T, K]
        v:  [B, H, T, V]
        g:  [B, H, T, K]  log-space per-dim forget gate (negative values)
        beta: [B, H, T, 1]  learning rate (sigmoid output, 0-1)
        initial_state: [B, H, K, V] initial recurrent state
        output_final_state: whether to return final state

    Returns:
        o: [B, H, T, V]  output
        final_state: [B, H, K, V] or None
    """
    dtype = v.dtype
    B, H, T, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5

    # L2 normalize q and k, then scale q
    q = F.normalize(q.float(), dim=-1) * scale
    k = F.normalize(k.float(), dim=-1)
    v = v.float()

    S = q.new_zeros(B, H, K, V, dtype=torch.float32)
    o = torch.zeros_like(v)

    if initial_state is not None:
        S = S + initial_state.float()

    for i in range(T):
        q_i = q[:, :, i]       # [B, H, K]
        k_i = k[:, :, i]       # [B, H, K]
        v_i = v[:, :, i]       # [B, H, V]
        g_i = g[:, :, i]       # [B, H, K] log-space gate
        beta_i = beta[:, :, i] # [B, H, 1]

        # Decay: S *= exp(g_i) per-dim
        S = S * g_i[..., :, None].exp()
        # Delta correction: v_error = v - k^T @ S
        # k_i^T @ S: [B, H, K] @ [B, H, K, V] -> [B, H, V]
        kS = (k_i[..., :, None] * S).sum(-2)  # [B, H, V]
        v_error = v_i - kS                      # [B, H, V]
        # Update: S += beta * k ⊗ v_error
        S = S + beta_i[..., None] * (k_i[..., :, None] * v_error[..., None, :])
        # Output: o = q^T @ S
        o[:, :, i] = (q_i[..., :, None] * S).sum(-2)

    final_state = S if output_final_state else None
    return o.to(dtype), final_state
