"""TTT-E2E sliding-window attention (L2).

Mirrors the JAX/Equinox reference at
``github.com/test-time-training/e2e:ttt/model/attention.py``:

  - 4 dense projections wq, wk, wv, wo (no bias, hidden -> hidden)
  - per-head Q/K RMSNorm on the head_dim axis (qk_norm=True in the paper)
  - interleaved RoPE (consecutive pairs are ``(real, imag)``) with
    LOCAL window-relative positions on the suffix path and GLOBAL token
    positions on the prefix path
  - sliding-window causal attention with window W (paper default: 8192)
  - prefix path: full sequence at once, no KV cache update
  - suffix path: chunked, attends each chunk over a (W,)-long rolling cache

Two forward entry points:
    forward_prefix(hidden_states, position_ids) -> hidden_states
    forward_suffix(hidden_states, kv_cache) -> hidden_states, new_kv_cache

The kb-nano L1 ``RotaryEmbedding`` is NeOX/half-split-style; the JAX
reference uses the GPT-J/interleaved style. We implement the interleaved
formula inline here (small, easy to verify) instead of pulling in a new
L1 op, keeping the L2 boundary clean.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..L1.linear import Linear
from ..L1.rms_norm import RMSNorm


def _rms_native_swa(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure-PyTorch RMSNorm matching the JAX reference's promote_dtype path.

    Used here instead of the kb-nano L1 RMSNorm CUDA kernel because that
    kernel is incorrect for the small head_dim (e.g. 16) seen in tiny test
    configs and has no torch.func/autograd backward registered. Computes
    in fp32, returns ``x.dtype``.
    """
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return ((xf * torch.rsqrt(var + eps)).to(orig) * weight)


def _precompute_freqs_cis(dim: int, end: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (end, dim/2) in fp32.

    Matches JAX:
        freqs = 1 / (theta ** (arange(0, dim, 2) / dim))   # (D/2,)
        t     = arange(end)                                # (T,)
        outer = t[:, None] * freqs[None, :]                # (T, D/2)
        cos, sin = cos(outer), sin(outer)
    """
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[:half] / dim))
    t = torch.arange(end, dtype=torch.float32)
    outer = torch.outer(t, freqs)
    return outer.cos(), outer.sin()


def _apply_rotary_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved RoPE.

    Args:
        x:   (B, T, H, D) — query or key, with D=head_dim
        cos: (T, D/2)
        sin: (T, D/2)
    """
    orig_dtype = x.dtype
    # x reshape: (B, T, H, D) -> (B, T, H, D/2, 2)
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x0 = x_pairs[..., 0]                                # (B, T, H, D/2)
    x1 = x_pairs[..., 1]
    # Reshape cos/sin to (1, T, 1, D/2) so they broadcast against (B, T, H, D/2).
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.stack([o0, o1], dim=-1).flatten(-2).to(orig_dtype)


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
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # We pre-compute RoPE once to cover both prefix (global positions up to
        # max_position_embeddings) and suffix (local positions 0..W+C-1).
        rope_len = max(max_position_embeddings, window_size + chunk_size)
        cos, sin = _precompute_freqs_cis(self.head_dim, rope_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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
        # RMSNorm over last (head_dim) axis. Matches JAX which vmaps a
        # head_dim-shaped RMSNorm over (heads, seq). We use _rms_native_swa
        # rather than the L1 CUDA kernel for two reasons: (1) the kernel
        # silently produces wrong output on 4D inputs at small head_dims
        # (verified empirically at head_dim=16), and (2) we want fp32-internal
        # math to match JAX's promote_dtype semantics regardless of param/
        # input dtype.
        return (
            _rms_native_swa(xq, self.q_norm.weight, self.q_norm.eps),
            _rms_native_swa(xk, self.k_norm.weight, self.k_norm.eps),
        )

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
        cos = self.rope_cos[position_ids].to(x.dtype)
        sin = self.rope_sin[position_ids].to(x.dtype)
        xq = _apply_rotary_interleaved(xq, cos, sin)
        xk = _apply_rotary_interleaved(xk, cos, sin)

        # (B, H, T, D) for SDPA
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # PyTorch SDPA supports causal mask but not sliding window directly.
        # Build a (T, T) bool mask: position i attends to j iff j <= i and i - j < W.
        i = torch.arange(t, device=x.device).unsqueeze(1)
        j = torch.arange(t, device=x.device).unsqueeze(0)
        attn_mask = (j <= i) & (i - j < self.window_size)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(b, t, self.hidden_size)
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
                holding POST-qk-norm, PRE-RoPE k/v from prior chunks.
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
        assert c == C
        device = x.device

        xq, xk, xv = self._project_qkv(x)
        xq, xk = self._qk_norm(xq, xk)

        # JAX reference stores the post-qk-norm but PRE-RoPE k/v in the cache.
        # The cache is the SOURCE of input k/v for the next chunk's attention,
        # at which point they get RoPE'd with shifted (older) positions.
        full_k_pre = torch.cat([k_cache, xk], dim=1)  # (B, W+C, H, D), pre-rope
        full_v = torch.cat([v_cache, xv], dim=1)

        q_pos = torch.arange(W, W + C, device=device)
        k_pos = torch.arange(W + C, device=device)
        q_cos = self.rope_cos[q_pos].to(x.dtype)
        q_sin = self.rope_sin[q_pos].to(x.dtype)
        k_cos = self.rope_cos[k_pos].to(x.dtype)
        k_sin = self.rope_sin[k_pos].to(x.dtype)
        xq = _apply_rotary_interleaved(xq, q_cos, q_sin)
        full_k = _apply_rotary_interleaved(full_k_pre, k_cos, k_sin)

        # Attention mask: matches JAX sw_causal_mask exactly.
        # Query has C rows at local positions [W, W+C-1].
        # Key has W+C cols at local positions [0, W+C-1].
        # Constraints (from JAX):
        #   qi >= ki          (causal)
        #   qi <  ki + W      (within window)
        #   ki >= 0           (always true here, but JAX guards against it)
        # Plus: keys in slots [0, W - chunk_id*C) are "ahead of population" (chunk_id*C
        # tokens have been written, padded with zeros). These should be masked out.
        # JAX uses the same `qi >= ki` and `ki >= 0` along with chunk_id to translate
        # local indices into global indices for the mask. The effective rule:
        #   global_qi = chunk_id*C + (qi - W) = chunk_id*C + 0..C-1
        #   global_ki = chunk_id*C + (ki - W)
        # so global_ki ranges over [(chunk_id-1)*C - (W-C), (chunk_id+1)*C - 1] =
        #   [chunk_id*C - W, chunk_id*C + C - 1]
        # Constraints in JAX (with chunk_id):
        #   starting_query_idx = chunk_id * C
        #   ending_query_idx   = starting_query_idx + C
        #   ending_key_idx     = ending_query_idx
        #   qi = arange(0, C) + starting_query_idx           shape (C, 1)
        #   ki = arange(-W-C, 0) + ending_key_idx            shape (1, W+C)
        #   mask = (qi >= ki) & (qi < ki + W) & (ki >= 0)
        starting_q = chunk_id * C
        qi = (torch.arange(C, device=device) + starting_q).unsqueeze(1)
        ki = (torch.arange(-(W + C), 0, device=device) + (starting_q + C)).unsqueeze(0)
        mask = (qi >= ki) & (qi < ki + W) & (ki >= 0)
        attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, C, W+C)

        # SDPA with explicit mask.
        xq_t = xq.transpose(1, 2)             # (B, H, C, D)
        xk_t = full_k.transpose(1, 2)          # (B, H, W+C, D)
        xv_t = full_v.transpose(1, 2)          # (B, H, W+C, D)
        out = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(b, C, self.hidden_size)
        out = self.wo(out)

        # New cache: keep the last W (pre-rope, post-qk-norm) k/v.
        # NB: we use full_k_pre (pre-rope) and full_v.
        new_k = full_k_pre[:, -W:].contiguous()
        new_v = full_v[:, -W:].contiguous()
        return out, (new_k, new_v)
