"""Multi-rate Hyper-Connections (mHC) primitives for DeepSeek V4.

DeepSeek V4 replaces the usual ``residual = x + sublayer(norm(x))`` residual
stream with *hyper-connections*: the hidden state is widened to ``hc_mult``
parallel streams and, at every sub-block boundary, the streams are mixed by a
learned, Sinkhorn-normalized routing matrix.  Each boundary is two ops:

* ``mhc_pre``  — RMS-norm the stacked streams, project them through ``fn`` to
  produce per-stream *pre* mix weights, a *post* mix vector, and a Sinkhorn-
  normalized ``hc_mult x hc_mult`` *comb* matrix, then collapse the streams
  into a single ``[num_tokens, hidden]`` layer input.
* ``mhc_post`` — recombine the sublayer output ``x`` with the ``hc_mult``
  residual streams using the *post* / *comb* mixes from ``mhc_pre``.

The model also has a final ``hc_head`` that collapses the streams before the
LM head.

These wrap the **exact** compiled kernels vLLM ships in its 0.20.0 wheel:
``vllm.model_executor.layers.mhc.mhc_pre`` / ``mhc_post`` (TileLang) and the
``tf32_hc_prenorm_gemm`` GEMM from the vendored DeepGEMM.  Re-using the wheel
kernels guarantees bit-for-bit parity with the reference; see
``vllm/model_executor/layers/mhc.py`` and ``vllm/model_executor/models/
deepseek_v4.py`` (``hc_head``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHCPre(nn.Module):
    """mHC pre-block mix.

    Mirrors ``DeepseekV4DecoderLayer.hc_pre`` -> ``torch.ops.vllm.mhc_pre``.

    forward args (all tensor / scalar, custom-op safe):
        residual: (num_tokens, hc_mult, hidden), bfloat16
        fn:       (hc_mult*(2+hc_mult), hc_mult*hidden), float32
        hc_scale: (3,), float32
        hc_base:  (hc_mult*(2+hc_mult),), float32

    Returns ``(layer_input, post_mix, comb_mix)`` matching the model's
    ``hc_pre`` return order:
        layer_input: (num_tokens, hidden), bfloat16
        post_mix:    (num_tokens, hc_mult, 1), float32
        comb_mix:    (num_tokens, hc_mult, hc_mult), float32
    """

    def __init__(self, rms_eps: float, hc_pre_eps: float,
                 hc_sinkhorn_eps: float, hc_post_mult_value: float,
                 sinkhorn_iters: int):
        super().__init__()
        self.rms_eps = rms_eps
        self.hc_pre_eps = hc_pre_eps
        self.hc_sinkhorn_eps = hc_sinkhorn_eps
        self.hc_post_mult_value = hc_post_mult_value
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, residual: torch.Tensor, fn: torch.Tensor,
                hc_scale: torch.Tensor, hc_base: torch.Tensor):
        # Registers torch.ops.vllm.mhc_pre / mhc_post on import.
        import vllm.model_executor.layers.mhc  # noqa: F401

        post_mix, comb_mix, layer_input = torch.ops.vllm.mhc_pre(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_eps,
            hc_pre_eps=self.hc_pre_eps,
            hc_sinkhorn_eps=self.hc_sinkhorn_eps,
            hc_post_mult_value=self.hc_post_mult_value,
            sinkhorn_repeat=self.sinkhorn_iters,
        )
        return layer_input, post_mix, comb_mix


class MHCPost(nn.Module):
    """mHC post-block recombination.

    Mirrors ``DeepseekV4DecoderLayer.hc_post`` -> ``torch.ops.vllm.mhc_post``.

        x:        (num_tokens, hidden), bfloat16   (sublayer output)
        residual: (num_tokens, hc_mult, hidden), bfloat16
        post:     (num_tokens, hc_mult, 1), float32
        comb:     (num_tokens, hc_mult, hc_mult), float32
    Returns the updated stream stack (num_tokens, hc_mult, hidden), bfloat16.
    """

    def forward(self, x: torch.Tensor, residual: torch.Tensor,
                post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        import vllm.model_executor.layers.mhc  # noqa: F401
        return torch.ops.vllm.mhc_post(x, residual, post, comb)


@torch.compile(dynamic=True)
def _hc_head_impl(hidden_states: torch.Tensor, hc_fn: torch.Tensor,
                  hc_scale: torch.Tensor, hc_base: torch.Tensor,
                  rms_norm_eps: float, hc_eps: float) -> torch.Tensor:
    """Verbatim port of ``deepseek_v4.hc_head`` (collapse streams pre-LM-head)."""
    x = hidden_states
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


class HCHead(nn.Module):
    """Final hyper-connection collapse before the LM head / final norm."""

    def __init__(self, rms_norm_eps: float, hc_eps: float):
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.hc_eps = hc_eps

    def forward(self, hidden_states: torch.Tensor, hc_fn: torch.Tensor,
                hc_scale: torch.Tensor, hc_base: torch.Tensor) -> torch.Tensor:
        return _hc_head_impl(hidden_states, hc_fn, hc_scale, hc_base,
                             self.rms_norm_eps, self.hc_eps)
