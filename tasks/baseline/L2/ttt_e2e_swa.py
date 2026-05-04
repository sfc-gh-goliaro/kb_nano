"""TTT-E2E sliding-window attention (L2).

Mirrors the JAX/Equinox reference at
``github.com/test-time-training/e2e:ttt/model/attention.py``:

  - 4 dense projections wq, wk, wv, wo (no bias, hidden -> hidden)
  - per-head Q/K RMSNorm on the head_dim axis (qk_norm=True in the paper)
  - interleaved RoPE with LOCAL window-relative positions on the suffix
    path and GLOBAL token positions on the prefix path
  - sliding-window causal attention with window W (paper default: 8192)
  - prefix path: full sequence at once, no KV cache update
  - suffix path: chunked, attends each chunk over a (W,)-long rolling cache

This file is composed entirely of L1 ops (no torch.nn / torch.nn.functional
calls). It uses:
  - L1 :class:`Linear` for the four projections
  - L1 :class:`RMSNormNative` for q/k norm (autograd-friendly + correct on
    odd head_dim values where the L1 RMSNorm CUDA kernel is broken)
  - L1 :class:`TTTE2ERoPE` for interleaved RoPE
  - L1 :class:`DenseAttention` for the SDPA call (handles both prefix
    and suffix paths via the ``attn_mask`` argument; the kb-nano CUDA
    flash-attention L1 ops are paged-KV-cache-only and don't fit either
    of TTT-E2E's access patterns).
"""

from __future__ import annotations

import torch
from torch import nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.rms_norm_native import RMSNormNative
from ..L1.ttt_e2e_rope import TTTE2ERoPE


class TTTE2ESWA(nn.Module):
    """Sliding-window attention with prefix and suffix-chunked KV-cache paths.

    Args:
        hidden_size: model dim
        num_heads:   number of attention heads (head_dim = hidden_size // num_heads)
        window_size: sliding window size W (paper: 8192)
        chunk_size:  mini-batch tokens per inner-loop step (paper: 1024)
        max_position_embeddings: how many RoPE positions to precompute (must be
            >= max(seq_len, W + chunk_size))
        rope_theta: RoPE theta (paper: 500000)
        qk_norm: apply per-head RMSNorm to Q and K (paper: True)
        rms_norm_eps: eps for the q/k RMSNorms
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        chunk_size: int,
        max_position_embeddings: int,
        rope_theta: float = 500000.0,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.qk_norm = qk_norm

        self.wq = Linear(hidden_size, hidden_size, bias=False)
        self.wk = Linear(hidden_size, hidden_size, bias=False)
        self.wv = Linear(hidden_size, hidden_size, bias=False)
        self.wo = Linear(hidden_size, hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNormNative(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNormNative(self.head_dim, eps=rms_norm_eps)

        rope_len = max(max_position_embeddings, window_size + chunk_size)
        self.rope = TTTE2ERoPE(self.head_dim, rope_len, rope_theta=rope_theta)

        # SDPA dispatcher (auto-selects flash on Hopper, SDPA elsewhere).
        # We always pass attn_mask, which forces the SDPA fallback path —
        # acceptable here because the windowed mask is small (chunk_size by
        # window+chunk_size keys) and the TTT-E2E inner loop is not the
        # attention bottleneck anyway.
        self.attn = DenseAttention(backend="sdpa")

        # Cache for the chunk-suffix attention masks and the RoPE position
        # vectors. All keyed by ``device`` so we lazily materialize on the
        # right GPU on first call. Mask cache is also per-chunk_id since
        # the mask depends on chunk_id (early chunks have masked-out cache
        # padding via ``ki >= 0``); position vectors are chunk-invariant.
        self._suffix_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self._suffix_q_pos: dict[torch.device, torch.Tensor] = {}
        self._suffix_k_pos: dict[torch.device, torch.Tensor] = {}

    # ------------------------------------------------------------------ helpers

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to (xq, xk, xv) of shape (B, T, H, D)."""
        b, t, _ = x.shape
        xq = self.wq(x).view(b, t, self.num_heads, self.head_dim)
        xk = self.wk(x).view(b, t, self.num_heads, self.head_dim)
        xv = self.wv(x).view(b, t, self.num_heads, self.head_dim)
        return xq, xk, xv

    def _qk_norm(self, xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.qk_norm:
            return xq, xk
        # RMSNormNative handles per-head_dim normalization on (B, T, H, D)
        # directly — the last-dim broadcast is correct and dtype-promoting.
        return self.q_norm(xq), self.k_norm(xk)

    # --------------------------------------------------------------- prefix path

    def forward_prefix(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Full-sequence sliding-window attention. No KV cache update.

        Args:
            x: (B, T, hidden)
            position_ids: (T,) int positions for RoPE. If None, uses 0..T-1.
        """
        b, t, _ = x.shape
        xq, xk, xv = self._project_qkv(x)
        xq, xk = self._qk_norm(xq, xk)

        if position_ids is None:
            position_ids = torch.arange(t, device=x.device)
        xq = self.rope(xq, position_ids)
        xk = self.rope(xk, position_ids)

        # Sliding causal window: position i attends to j iff j <= i and i - j < W.
        i = torch.arange(t, device=x.device).unsqueeze(1)
        j = torch.arange(t, device=x.device).unsqueeze(0)
        attn_mask = (j <= i) & (i - j < self.window_size)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # DenseAttention takes (B, T, H, D), softmax_scale, causal, attn_mask.
        out = self.attn(xq, xk, xv, softmax_scale=self.head_dim ** -0.5,
                        causal=False, attn_mask=attn_mask)
        out = out.reshape(b, t, self.hidden_size)
        return self.wo(out)

    # --------------------------------------------------------------- suffix path

    def init_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.zeros(batch_size, self.window_size, self.num_heads, self.head_dim, dtype=dtype, device=device)
        v = torch.zeros_like(k)
        return k, v

    def forward_suffix_chunk(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        chunk_id: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Chunked sliding-window attention with rolling KV cache.

        Args:
            x: (B, C, hidden) where C == chunk_size
            kv_cache: (k_cache, v_cache) each of shape (B, W, H, D),
                holding POST-qk-norm, PRE-RoPE k/v from prior chunks
                (matches the JAX reference's storage layout).
            chunk_id: 0-indexed chunk number within the current sequence;
                used to mask out cache slots that haven't been written yet.
        Returns:
            out:        (B, C, hidden)
            new_kv_cache: (k_cache', v_cache') — last W (post-qk-norm,
                pre-RoPE) k/v values from cat(cache, new_chunk).
        """
        b, c, _ = x.shape
        assert c == self.chunk_size, f"chunk size mismatch: got {c}, expected {self.chunk_size}"
        k_cache, v_cache = kv_cache
        W = self.window_size
        C = self.chunk_size
        device = x.device

        xq, xk, xv = self._project_qkv(x)
        xq, xk = self._qk_norm(xq, xk)

        # Cache stores POST-qk-norm but PRE-RoPE k/v. Concat with new chunk's
        # (post-qk-norm, pre-RoPE) k/v, then RoPE the cat'd tensor at LOCAL
        # window positions [0..W+C-1] and the new q at [W..W+C-1].
        full_k_pre = torch.cat([k_cache, xk], dim=1)          # (B, W+C, H, D)
        full_v = torch.cat([v_cache, xv], dim=1)

        q_pos = self._suffix_q_pos.get(device)
        if q_pos is None:
            q_pos = torch.arange(W, W + C, device=device)
            self._suffix_q_pos[device] = q_pos
        k_pos = self._suffix_k_pos.get(device)
        if k_pos is None:
            k_pos = torch.arange(W + C, device=device)
            self._suffix_k_pos[device] = k_pos
        xq = self.rope(xq, q_pos)
        full_k = self.rope(full_k_pre, k_pos)

        # JAX sw_causal_mask: query index in (chunk_id*C, chunk_id*C + C),
        # key index in (chunk_id*C - W, chunk_id*C + C - 1). Constraints:
        #   qi >= ki, qi < ki + W, ki >= 0.
        # The mask is fully determined by (chunk_id, C, W) and is reused
        # across every chunk-by-chunk call at the same chunk_id, so we
        # cache it to avoid a handful of Python+GPU launches per call.
        cache_key = (chunk_id, device)
        attn_mask = self._suffix_mask_cache.get(cache_key)
        if attn_mask is None:
            starting_q = chunk_id * C
            qi = (torch.arange(C, device=device) + starting_q).unsqueeze(1)
            ki = (torch.arange(-(W + C), 0, device=device) + (starting_q + C)).unsqueeze(0)
            mask = (qi >= ki) & (qi < ki + W) & (ki >= 0)
            attn_mask = mask.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, C, W+C)
            self._suffix_mask_cache[cache_key] = attn_mask

        out = self.attn(xq, full_k, full_v,
                        softmax_scale=self.head_dim ** -0.5,
                        causal=False, attn_mask=attn_mask)
        out = out.reshape(b, C, self.hidden_size)
        out = self.wo(out)

        new_k = full_k_pre[:, -W:].contiguous()
        new_v = full_v[:, -W:].contiguous()
        return out, (new_k, new_v)
