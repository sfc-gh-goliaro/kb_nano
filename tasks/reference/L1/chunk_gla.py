"""Semantic PyTorch reference for chunk_gla.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

The production kernel is FLA's chunked GLA prefill: the sequence is processed
in fixed-size chunks with an inter-chunk fp32 state recurrence plus an
intra-chunk attention matrix.  A per-token recurrence is NOT an adequate
reference here: the chunk algorithm rounds the decayed keys/queries and the
per-chunk states to the input dtype before its matmuls, and the final state is
compared under float32 tolerances, so the reference must follow the same
chunked structure and the same rounding points to agree numerically.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** ((n - 1).bit_length())


def _gla_intra_A(
    q_c: torch.Tensor,   # (N, L, H, K) input dtype
    k_c: torch.Tensor,   # (N, L, H, K) input dtype
    gc: torch.Tensor,    # (N, L, H, K) fp32 chunk-local inclusive cumsum of g
    scale: float,
    sub_block: int = 16,
) -> torch.Tensor:
    """Intra-chunk attention matrix A (fp32), lower-triangular incl. diagonal.

    A[l, j] = scale * sum_k q[l, k] * k[j, k] * exp(gc[l, k] - gc[j, k]) for
    j <= l, else 0.  Computed in sub-blocks anchored at the sub-block's first
    row (``gn``) so every exponent is <= 0 and cannot overflow, matching the
    production kernel's factorization.
    """
    N, L, H, K = q_c.shape
    A = q_c.new_zeros((N, H, L, L), dtype=torch.float32)
    for i0 in range(0, L, sub_block):
        i1 = min(i0 + sub_block, L)
        gn = gc[:, i0]                      # (N, H, K) anchor row
        qi = q_c[:, i0:i1].float()          # (N, Li, H, K)
        gi = gc[:, i0:i1]
        if i0 > 0:
            # blocks strictly below the diagonal: factored matmul form
            qg = qi * torch.exp(gi - gn[:, None]) * scale
            kg = k_c[:, :i0].float() * torch.exp(gn[:, None] - gc[:, :i0])
            A[:, :, i0:i1, :i0] = torch.einsum("nlhk,njhk->nhlj", qg, kg)
        # diagonal block: elementwise, masked before exp (j <= l only)
        kj = k_c[:, i0:i1].float()
        gj = gc[:, i0:i1]
        li = i1 - i0
        diff = gi[:, :, None] - gj[:, None, :]          # (N, Li, Lj, H, K)
        tri = (torch.arange(li, device=q_c.device)[:, None]
               >= torch.arange(li, device=q_c.device)[None, :])
        diff = torch.where(tri[None, :, :, None, None], diff,
                           torch.full_like(diff, float("-inf")))
        Ad = (qi[:, :, None] * kj[:, None] * torch.exp(diff)).sum(-1) * scale
        A[:, :, i0:i1, i0:i1] = Ad.permute(0, 3, 1, 2)
    return A


def _gla_chunk_step(
    q_c: torch.Tensor,   # (N, L, H, K)
    k_c: torch.Tensor,   # (N, L, H, K)
    v_c: torch.Tensor,   # (N, L, H, V)
    gc: torch.Tensor,    # (N, L, H, K) fp32 chunk-local cumsum
    h: torch.Tensor,     # (N, H, K, V) fp32 running state
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunk of the forward: returns (o_chunk, new_state).

    Rounding points mirror the production kernel: the state consumed by the
    output matmul and the decayed q/k entering matmuls are rounded to the
    input dtype; the state recurrence itself accumulates in fp32.
    """
    dtype = v_c.dtype
    h_b = h.to(dtype)                                    # per-chunk stored state
    qg = (q_c.float() * torch.exp(gc)).to(dtype)
    o = torch.einsum("nlhk,nhkv->nlhv", qg.float(), h_b.float()) * scale
    A = _gla_intra_A(q_c, k_c, gc, scale).to(dtype)
    o = o + torch.einsum("nhlj,njhv->nlhv", A.float(), v_c.float())

    g_last = gc[:, -1]                                   # (N, H, K)
    kg = (k_c.float() * torch.exp(g_last[:, None] - gc)).to(dtype)
    h = h * torch.exp(g_last)[..., None] + torch.einsum(
        "nlhk,nlhv->nhkv", kg.float(), v_c.float())
    return o.to(dtype), h


def _chunk_gla_one(
    q: torch.Tensor,     # (N, T, H, K)
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    h: torch.Tensor,     # (N, H, K, V) fp32
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = q.shape[1]
    o = torch.zeros_like(v)
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        gc = g[:, s:e].float().cumsum(1)                 # chunk-local, fp32
        o[:, s:e], h = _gla_chunk_step(
            q[:, s:e], k[:, s:e], v[:, s:e], gc, h, scale)
    return o, h


class ChunkGLA(nn.Module):
    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        g: torch.Tensor,  # [B, T, H, K]  log-space forget gate
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

        if cu_seqlens is None:
            h = (initial_state.float().clone() if initial_state is not None
                 else q.new_zeros((B, H, K, V), dtype=torch.float32))
            o, h = _chunk_gla_one(q, k, v, g, scale, h, chunk_size)
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
            o[:, s:e], h = _chunk_gla_one(
                q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e],
                scale, h, chunk_size)
            ht[n] = h[0]
        return o, (ht if output_final_state else None)
