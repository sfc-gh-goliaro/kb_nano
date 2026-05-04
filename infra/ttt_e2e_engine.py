"""TTT-E2E inference engine.

Wraps :class:`tasks.baseline.L4.ttt_e2e.TTTE2EPipeline` with a state-dict
loader that reads the JAX-reference's portable ``.npz`` (produced by
``tests/bench_ttt_e2e_jax_worker.py init_and_save``). The engine is
deliberately small — all model logic lives in L1-L4; this file is wiring.

Layout expected in the npz (path = JAX equinox tree path joined by dots):

  language_model.model.wte.weight                                       (V, H)
  language_model.model.ln_f.weight                                      (H,)
  language_model.model.h.blocks.seq_norm.weight                         (L, H)
  language_model.model.h.blocks.ffn_norm.weight                         (L, H)
  language_model.model.h.blocks.seq_post_norm.weight                    (L, H)
  language_model.model.h.blocks.ffn_post_norm.weight                    (L, H)
  language_model.model.h.blocks.seq_modeling_block.{wq,wk,wv,wo}.weight (L, H, H)
  language_model.model.h.blocks.seq_modeling_block.{q,k}_norm.weight    (L, D)  D=head_dim
  language_model.model.h.blocks.feed_forward.{w1,w3}.weight             (L, H, I)
  language_model.model.h.blocks.feed_forward.w2.weight                  (L, I, H)
  language_model.model.h.prime_storage.feed_forward_prime.{w1,w3}.weight (S, H, I)
  language_model.model.h.prime_storage.feed_forward_prime.w2.weight     (S, I, H)
  language_model.model.h.prime_storage.{ffn_prime_norm,ffn_prime_post_norm}.weight (S, H)

JAX ``NormalLinear`` weight shape is ``(in, out)`` so ``y = x @ W``, while
PyTorch ``nn.Linear`` weight is ``(out, in)`` (``F.linear`` does ``y =
x @ W.T``). The mapping below transposes accordingly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..tasks.baseline.L4.ttt_e2e import TTTE2EConfig, TTTE2EPipeline

logger = logging.getLogger(__name__)


_JAX_PREFIX = "language_model.model"


def _bn(prefix: str) -> str:
    return f"{_JAX_PREFIX}.h.blocks.{prefix}"


def _pn(prefix: str) -> str:
    return f"{_JAX_PREFIX}.h.prime_storage.{prefix}"


def _to_torch(arr: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.to(device=device, dtype=dtype)


def load_jax_npz_into_pipeline(
    pipeline: TTTE2EPipeline,
    npz_path: str | Path,
    *,
    device: torch.device | None = None,
    param_dtype: torch.dtype | None = None,
) -> None:
    """Load a JAX-reference ``.npz`` of trainable weights into ``pipeline``.

    Mutates the pipeline's parameters in-place. ``param_dtype`` defaults to
    the pipeline's configured ``param_dtype`` (fp32, matching JAX). The pipeline
    can subsequently be cast to bf16 compute via ``.to(dtype=...)`` if desired,
    though for parity with the reference we typically keep params in fp32.
    """
    cfg = pipeline.config
    if device is None:
        device = next(pipeline.parameters()).device
    if param_dtype is None:
        param_dtype = cfg.param_dtype

    z = np.load(str(npz_path))
    keys = set(z.keys())

    L = cfg.num_hidden_layers
    S = cfg.suffix_len
    H = cfg.hidden_size
    head_dim = H // cfg.num_attention_heads

    def _need(k: str) -> np.ndarray:
        if k not in keys:
            raise KeyError(f"missing key in npz: {k!r}")
        return z[k]

    n_loaded = 0

    # ---- token embedding (tied)
    wte_arr = _need(f"{_JAX_PREFIX}.wte.weight")  # (V, H), JAX uses x = embed(idx)
    pipeline.model.wte.emb.weight.data.copy_(_to_torch(wte_arr, param_dtype, device))
    n_loaded += 1

    # ---- final norm
    ln_f_arr = _need(f"{_JAX_PREFIX}.ln_f.weight")
    pipeline.model.ln_f.weight.data.copy_(_to_torch(ln_f_arr, param_dtype, device))
    n_loaded += 1

    # ---- per-layer params (vmapped in JAX, leading dim = num_layers)
    seq_norm = _need(_bn("seq_norm.weight"))
    seq_post_norm = _need(_bn("seq_post_norm.weight"))
    ffn_norm = _need(_bn("ffn_norm.weight"))
    ffn_post_norm = _need(_bn("ffn_post_norm.weight"))
    wq = _need(_bn("seq_modeling_block.wq.weight"))   # (L, H, H), JAX stores (in, out)
    wk = _need(_bn("seq_modeling_block.wk.weight"))
    wv = _need(_bn("seq_modeling_block.wv.weight"))
    wo = _need(_bn("seq_modeling_block.wo.weight"))
    q_norm = _need(_bn("seq_modeling_block.q_norm.weight"))   # (L, D)
    k_norm = _need(_bn("seq_modeling_block.k_norm.weight"))
    ff_w1 = _need(_bn("feed_forward.w1.weight"))      # (L, H, I)
    ff_w2 = _need(_bn("feed_forward.w2.weight"))      # (L, I, H)
    ff_w3 = _need(_bn("feed_forward.w3.weight"))      # (L, H, I)

    for i in range(L):
        layer = pipeline.model.layers[i]
        layer.seq_norm.weight.data.copy_(_to_torch(seq_norm[i], param_dtype, device))
        layer.seq_post_norm.weight.data.copy_(_to_torch(seq_post_norm[i], param_dtype, device))
        layer.ffn_norm.weight.data.copy_(_to_torch(ffn_norm[i], param_dtype, device))
        layer.ffn_post_norm.weight.data.copy_(_to_torch(ffn_post_norm[i], param_dtype, device))
        # Linear(in, out).weight has shape (out, in); JAX has (in, out) — transpose.
        layer.seq_modeling_block.wq.weight.data.copy_(_to_torch(wq[i].T, param_dtype, device))
        layer.seq_modeling_block.wk.weight.data.copy_(_to_torch(wk[i].T, param_dtype, device))
        layer.seq_modeling_block.wv.weight.data.copy_(_to_torch(wv[i].T, param_dtype, device))
        layer.seq_modeling_block.wo.weight.data.copy_(_to_torch(wo[i].T, param_dtype, device))
        if cfg.qk_norm:
            layer.seq_modeling_block.q_norm.weight.data.copy_(_to_torch(q_norm[i], param_dtype, device))
            layer.seq_modeling_block.k_norm.weight.data.copy_(_to_torch(k_norm[i], param_dtype, device))
        layer.feed_forward.w1.weight.data.copy_(_to_torch(ff_w1[i].T, param_dtype, device))
        layer.feed_forward.w2.weight.data.copy_(_to_torch(ff_w2[i].T, param_dtype, device))
        layer.feed_forward.w3.weight.data.copy_(_to_torch(ff_w3[i].T, param_dtype, device))
        n_loaded += 11 + (2 if cfg.qk_norm else 0)

    # ---- prime params (vmapped over suffix_len, populate the last S blocks)
    if cfg.has_prime and S > 0:
        pp_w1 = _need(_pn("feed_forward_prime.w1.weight"))
        pp_w2 = _need(_pn("feed_forward_prime.w2.weight"))
        pp_w3 = _need(_pn("feed_forward_prime.w3.weight"))
        pp_norm = _need(_pn("ffn_prime_norm.weight"))
        pp_post_norm = _need(_pn("ffn_prime_post_norm.weight"))
        for s in range(S):
            target = pipeline.model.layers[L - S + s]
            assert target.has_prime
            target.feed_forward_prime.w1.weight.data.copy_(_to_torch(pp_w1[s].T, param_dtype, device))
            target.feed_forward_prime.w2.weight.data.copy_(_to_torch(pp_w2[s].T, param_dtype, device))
            target.feed_forward_prime.w3.weight.data.copy_(_to_torch(pp_w3[s].T, param_dtype, device))
            target.ffn_prime_norm.weight.data.copy_(_to_torch(pp_norm[s], param_dtype, device))
            target.ffn_prime_post_norm.weight.data.copy_(_to_torch(pp_post_norm[s], param_dtype, device))
            n_loaded += 5

    logger.info(
        "Loaded %d tensors from %s into TTTE2EPipeline (L=%d, S=%d, H=%d).",
        n_loaded, npz_path, L, S, H,
    )


class TTTE2EEngine:
    """Lightweight engine: build pipeline, load weights, run forward."""

    def __init__(
        self,
        config: TTTE2EConfig,
        weights_npz: str | Path | None = None,
        *,
        device: str | torch.device = "cuda",
        param_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
        prime_dtype: torch.dtype = torch.float32,
    ):
        """
        ``param_dtype`` is the storage dtype of the bulk of model weights —
        bf16 by default, matching the JAX reference's ``compute_dtype="bf16"``
        during forward. The kb-nano L1 RMSNorm/Linear CUDA kernels require x
        and weight to share dtype, so storing weights in bf16 lets us hit the
        fast paths.

        ``prime_dtype`` is the dtype the inner-loop SGD works in (fp32 by
        default, matching the JAX reference's ``state_dtype="fp32"``). The
        pipeline upcasts the cloned prime params to ``prime_dtype`` at the
        start of each sequence; the bf16-stored values just seed the clone.
        """
        self.config = config
        self.device = torch.device(device)
        # Pipeline knobs: dtype = forward compute dtype; param_dtype = inner
        # SGD dtype. (The actual stored param dtype is ``param_dtype`` arg.)
        self.config.dtype = compute_dtype
        self.config.param_dtype = prime_dtype
        self.pipeline = TTTE2EPipeline(config).to(device=self.device, dtype=param_dtype)
        if weights_npz is not None:
            load_jax_npz_into_pipeline(
                self.pipeline, weights_npz, device=self.device, param_dtype=param_dtype,
            )
        self.pipeline.eval()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        train_mode: str = "meta",
    ):
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if target_tokens is not None and target_tokens.device != self.device:
            target_tokens = target_tokens.to(self.device)
        return self.pipeline(input_ids=input_ids, target_tokens=target_tokens, train_mode=train_mode)
