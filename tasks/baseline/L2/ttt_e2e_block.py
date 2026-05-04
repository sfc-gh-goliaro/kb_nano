"""TTT-E2E transformer block (L2).

Mirrors ``ttt/model/transformer.py:Block`` exactly. Layout per block:

  Note on RMSNorm dispatch: the kb-nano L1 RMSNorm prefers a CUDA kernel via
  ``torch.ops._C.rms_norm`` (in-place mutating custom op) for max perf in the
  default forward path. That custom op does NOT register a backward, so
  ``torch.func.grad`` returns zeros through it. When the L4 pipeline takes an
  inner-loop SGD step over the prime FFN params, every RMSNorm downstream of
  the prime FFN sits in the gradient path and must dispatch to a pure-PyTorch
  implementation. We expose a private ``_rms_native`` here that does exactly
  that, and use it in the prime-override branch of the suffix path.

    x0 = x
    x  = pre_norm(seq_norm) -> SWA -> post_norm(seq_post_norm)
    x  = x + x0

    [if has_prime (suffix blocks only):
        x0 = x
        x  = pre_norm(ffn_prime_norm) -> SwiGLU_prime -> post_norm(ffn_prime_post_norm)
        x  = x + x0
    ]

    x0 = x
    x  = pre_norm(ffn_norm) -> SwiGLU -> post_norm(ffn_post_norm)
    x  = x + x0

The "prime" FFN's weights are present only on the last ``suffix_len`` blocks,
and are test-time-trained chunk-by-chunk by the L4 pipeline (this module
just exposes them as parameters and applies them when present).

Two forward entry points:
    forward_prefix(x, position_ids) — full sequence, no KV cache update
    forward_suffix_chunk(x, kv_cache, chunk_id) — chunked with rolling cache
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..L1.rms_norm import RMSNorm
from .ttt_e2e_swa import TTTE2ESWA
from .ttt_e2e_swiglu import TTTE2ESwiGLU


def _rms_native(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure-PyTorch RMSNorm — matches the kb-nano L1 ``forward_native`` math.

    Used in inner-loop gradient paths where the CUDA custom op (no autograd
    registered) would silently drop gradients. Computes RMS in fp32 for
    numerical stability and returns the ``weight``-dtype result, mirroring
    the JAX reference's ``promote_dtype`` -> compute_dtype -> RMS -> weight
    pattern. Output dtype follows ``x.dtype`` after the multiply.
    """
    orig_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf.to(orig_dtype) * weight)


class TTTE2EBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        window_size: int,
        chunk_size: int,
        max_position_embeddings: int,
        rope_theta: float = 500000.0,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        has_prime: bool = False,
    ):
        super().__init__()
        self.has_prime = has_prime
        self.hidden_size = hidden_size

        self.seq_modeling_block = TTTE2ESWA(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            chunk_size=chunk_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            rms_norm_eps=rms_norm_eps,
        )
        self.feed_forward = TTTE2ESwiGLU(hidden_size, intermediate_size)

        self.seq_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.seq_post_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn_post_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        if has_prime:
            self.feed_forward_prime = TTTE2ESwiGLU(hidden_size, intermediate_size)
            self.ffn_prime_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.ffn_prime_post_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        else:
            self.feed_forward_prime = None
            self.ffn_prime_norm = None
            self.ffn_prime_post_norm = None

    # -------------------------------------------------------- shared FFN helper

    def _apply_ffn(self, x: torch.Tensor, ffn: nn.Module, pre: nn.Module, post: nn.Module) -> torch.Tensor:
        h = pre(x)
        h = ffn(h)
        h = post(h)
        return h

    def _apply_seq(
        self,
        x: torch.Tensor,
        is_prefix: bool,
        position_ids: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        chunk_id: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        h = self.seq_norm(x)
        if is_prefix:
            h = self.seq_modeling_block.forward_prefix(h, position_ids=position_ids)
            new_cache = None
        else:
            h, new_cache = self.seq_modeling_block.forward_suffix_chunk(h, kv_cache, chunk_id)
        h = self.seq_post_norm(h)
        return h, new_cache

    # ---------------------------------------------------------------- prefix path

    def forward_prefix(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self._apply_seq(x, is_prefix=True, position_ids=position_ids, kv_cache=None, chunk_id=0)
        x = x + h

        if self.has_prime:
            h = self._apply_ffn(x, self.feed_forward_prime, self.ffn_prime_norm, self.ffn_prime_post_norm)
            x = x + h

        h = self._apply_ffn(x, self.feed_forward, self.ffn_norm, self.ffn_post_norm)
        x = x + h
        return x

    # ---------------------------------------------------------------- suffix path

    def forward_suffix_chunk(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        chunk_id: int,
        prime_override: dict | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one chunk through this suffix block.

        ``prime_override``: optional dict ``{"w1": Tensor, "w2": Tensor,
        "w3": Tensor}`` of fresh prime-FFN weights. When provided, the prime
        FFN is called functionally with those weights instead of the stored
        Parameters; ALL norms downstream of the prime FFN dispatch to a
        pure-PyTorch RMSNorm so gradients flow through them under
        ``torch.func.grad``. (The L1 CUDA RMSNorm has no backward registered.)
        """
        h, new_cache = self._apply_seq(x, is_prefix=False, position_ids=None, kv_cache=kv_cache, chunk_id=chunk_id)
        x = x + h

        if self.has_prime:
            assert self.feed_forward_prime is not None
            grad_path = prime_override is not None

            # ffn_prime_norm: this norm is BEFORE the prime FFN — the prime
            # weights don't depend on its output, so grad doesn't flow back
            # through it. We can use the fast L1 path either way.
            if grad_path:
                h_in = _rms_native(x, self.ffn_prime_norm.weight, self.ffn_prime_norm.eps)
            else:
                h_in = self.ffn_prime_norm(x)

            if prime_override is None:
                h = self.feed_forward_prime(h_in)
            else:
                # Match the JAX reference: prime params live in state_dtype
                # (fp32) and are differentiated in fp32; the forward casts them
                # to compute_dtype (bf16) via promote_dtype right before the
                # matmul. ``v.to(h_in.dtype)`` is a differentiable cast, so the
                # SGD gradient path back to the fp32 ``prime_override`` is
                # preserved. We also bypass nn.Linear here in favour of bare
                # F.linear for the same reason: nn.Linear has no backward
                # under torch.func when used via functional_call on a custom
                # subclass, but F.linear is plain autograd-friendly.
                w1 = prime_override["w1"].to(h_in.dtype)
                w2 = prime_override["w2"].to(h_in.dtype)
                w3 = prime_override["w3"].to(h_in.dtype)
                z1 = F.linear(h_in, w1)
                z3 = F.linear(h_in, w3)
                h = F.linear(F.silu(z1) * z3, w2)

            if grad_path:
                h = _rms_native(h, self.ffn_prime_post_norm.weight, self.ffn_prime_post_norm.eps)
            else:
                h = self.ffn_prime_post_norm(h)
            x = x + h

        if prime_override is not None:
            # Regular FFN downstream of the prime path also sits in the grad
            # chain, so use pure-PyTorch math here too.
            h = _rms_native(x, self.ffn_norm.weight, self.ffn_norm.eps)
            ff = self.feed_forward
            z1 = F.linear(h, ff.w1.weight)
            z3 = F.linear(h, ff.w3.weight)
            h = F.linear(F.silu(z1) * z3, ff.w2.weight)
            h = _rms_native(h, self.ffn_post_norm.weight, self.ffn_post_norm.eps)
        else:
            h = self._apply_ffn(x, self.feed_forward, self.ffn_norm, self.ffn_post_norm)
        x = x + h
        return x, new_cache
