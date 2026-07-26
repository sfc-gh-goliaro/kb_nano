"""Semantic PyTorch reference for chunk_retention.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

The production kernel is FLA's chunked RetNet prefill (chunk_simple_gla with a
per-head data-independent decay ``gamma``).  The reference follows the same
chunked structure and the same rounding points: the per-chunk states and the
decayed intra-chunk attention matrix are rounded to the input dtype before
their matmuls, the state recurrence accumulates in fp32, and the final state
is returned in fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** ((n - 1).bit_length())


def _retention_gamma_log(num_heads: int, device) -> torch.Tensor:
    """Per-head log decay: log(1 - 2^(-5 - head_index)), fp32."""
    idx = torch.arange(num_heads, dtype=torch.float32, device=device)
    return (1.0 - torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=device), -5.0 - idx)).log()


def _retention_chunk_step(
    q_c: torch.Tensor,       # (N, L, H, K)
    k_c: torch.Tensor,       # (N, L, H, K)
    v_c: torch.Tensor,       # (N, L, H, V)
    gamma_log: torch.Tensor,  # (H,) fp32
    h: torch.Tensor,         # (N, H, K, V) fp32 running state
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = v_c.dtype
    L = q_c.shape[1]
    steps = torch.arange(1, L + 1, dtype=torch.float32, device=q_c.device)
    bg = gamma_log[None, :] * steps[:, None]             # (L, H): gamma * (i+1)
    g_last = gamma_log * float(L)                        # (H,)

    # output: inter-chunk part reads the bf16-rounded stored state
    h_b = h.to(dtype)
    o = torch.einsum("nlhk,nhkv->nlhv", q_c.float(), h_b.float())
    o = o * torch.exp(bg)[None, :, :, None]
    # intra-chunk attention: QK^T in fp32, then the gamma^(l-j) decay,
    # lower-triangular mask, rounded to the input dtype before A @ V
    A = torch.einsum("nlhk,njhk->nhlj", q_c.float(), k_c.float())
    decay = torch.exp(bg[:, None, :] - bg[None, :, :])   # (L, L, H)
    tri = (torch.arange(L, device=q_c.device)[:, None]
           >= torch.arange(L, device=q_c.device)[None, :])
    decay = torch.where(tri[:, :, None], decay, torch.zeros_like(decay))
    A = (A * decay.permute(2, 0, 1)[None]).to(dtype)
    o = o * scale + torch.einsum(
        "nhlj,njhv->nlhv", A.float(), v_c.float()) * scale

    # state recurrence: values decayed and rounded, fp32 accumulation
    vg = (v_c.float() * torch.exp(g_last[None, None, :] - bg[None])[..., None]
          ).to(dtype)
    h = h * torch.exp(g_last)[None, :, None, None] + torch.einsum(
        "nlhk,nlhv->nhkv", k_c.float(), vg.float())
    return o.to(dtype), h


class ChunkRetention(nn.Module):
    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [N, H, K, V] fp32
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, H, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K ** -0.5
        chunk_size = min(64, max(16, _next_pow2(T)))
        gamma_log = _retention_gamma_log(H, q.device)

        def run(qs, ks, vs, h):
            o = torch.zeros_like(vs)
            for s in range(0, qs.shape[1], chunk_size):
                e = min(s + chunk_size, qs.shape[1])
                o[:, s:e], h = _retention_chunk_step(
                    qs[:, s:e], ks[:, s:e], vs[:, s:e], gamma_log, h, scale)
            return o, h

        if cu_seqlens is None:
            h = (initial_state.float().clone() if initial_state is not None
                 else q.new_zeros((B, H, K, V), dtype=torch.float32))
            o, h = run(q, k, v, h)
            return o, (h if output_final_state else None)

        # varlen: B == 1, sequences packed along T
        num_seqs = cu_seqlens.numel() - 1
        o = torch.zeros_like(v)
        ht = q.new_zeros((num_seqs, H, K, V), dtype=torch.float32)
        for n in range(num_seqs):
            s = int(cu_seqlens[n].item())
            e = int(cu_seqlens[n + 1].item())
            if e <= s:
                continue
            h = (initial_state[n:n + 1].float().clone()
                 if initial_state is not None
                 else q.new_zeros((1, H, K, V), dtype=torch.float32))
            o[:, s:e], h = run(q[:, s:e], k[:, s:e], v[:, s:e], h)
            ht[n] = h[0]
        return o, (ht if output_final_state else None)
