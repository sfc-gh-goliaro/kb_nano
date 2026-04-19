"""FLA inference engine for recurrent linear-attention models.

Distinct from ``LlamaEngine`` because recurrent models (GLA / RetNet /
RWKV7) carry per-sequence state matrices instead of a paged KV cache.
The semantic gap is too wide to share scheduling code:

  - No paged KV cache: state is a per-layer ``[B, H, K, V]`` tensor.
  - No flat token layout: prefill and decode are batched ``[B, T, D]``
    forwards; padding and per-seq attention masks instead of varlen.
  - No CUDA-graph capture against fixed slot tables: the state buffers
    move whenever the active batch shape changes.

So we keep ``LlamaEngine`` clean for paged-KV models and put recurrent
scheduling here.

Surface area mirrors ``LlamaEngine`` for the bits ``bench_*.py`` and
user code touch:

  - ``FLAEngine(model_name, dtype, seed, max_num_seqs, ...).generate(prompts, sampling_params)``
  - ``SamplingParams`` and ``GenerationOutput`` are re-exported from
    ``infra.engine`` so callers can use either engine interchangeably.

Tensor parallel is not implemented (TP=1 only) — all FLA models we
target are < 10B params and fit on a single H200.
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass, field
from glob import glob
from typing import Any

import numpy as np
import torch

from .engine import GenerationOutput, SamplingParams  # re-exported

__all__ = ["FLAEngine", "SamplingParams", "GenerationOutput"]


# ---------------------------------------------------------------------------
# Per-sequence state container
# ---------------------------------------------------------------------------
@dataclass
class _ActiveSeq:
    """Bookkeeping for a single in-flight sequence."""
    seq_id: int
    prompt_ids: list[int]
    generated_ids: list[int] = field(default_factory=list)
    max_tokens: int = 512
    ignore_eos: bool = False
    sampling: SamplingParams = field(default_factory=SamplingParams)
    finished: bool = False
    # per-layer recurrent / conv state, keyed by id(L2 attention module)
    states: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_states: dict[int, torch.Tensor] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_REGISTRY = {
    "fla-hub/gla-2.7B-100B": ("kb_nano.tasks.baseline.L4.gla", "GLAConfig", "GLAForCausalLM"),
    "fla-hub/retnet-2.7B-100B": ("kb_nano.tasks.baseline.L4.retnet", "RetNetConfig", "RetNetForCausalLM"),
    "fla-hub/rwkv7-2.9B-g1": ("kb_nano.tasks.baseline.L4.rwkv7", "RWKV7Config", "RWKV7ForCausalLM"),
    "fla-hub/rwkv7-2.9B-world": ("kb_nano.tasks.baseline.L4.rwkv7", "RWKV7Config", "RWKV7ForCausalLM"),
}


def _load_model(model_name: str, dtype: torch.dtype, device: torch.device):
    """Load an FLA model via snapshot_download + safetensors weight copy.

    Returns ``(model, model_path)`` where ``model`` is on ``device`` /
    ``dtype`` and in eval mode.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import importlib

    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unsupported FLA model: {model_name!r}. "
            f"Known: {sorted(_REGISTRY)}"
        )

    module_name, config_cls, model_cls = _REGISTRY[model_name]
    mod = importlib.import_module(module_name)
    Config = getattr(mod, config_cls)
    Model = getattr(mod, model_cls)

    model_path = snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )
    config = Config.from_pretrained(model_path)
    model = Model(config)

    sf_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    loaded = 0
    for sf in sf_files:
        with safe_open(sf, "pt", "cpu") as f:
            for name in f.keys():
                try:
                    param = model.get_parameter(name)
                except AttributeError:
                    continue
                src = f.get_tensor(name)
                if src.shape != param.data.shape:
                    raise RuntimeError(
                        f"Shape mismatch loading {name}: "
                        f"checkpoint {tuple(src.shape)} vs model {tuple(param.data.shape)}"
                    )
                param.data.copy_(src)
                loaded += 1
    print(f"  [FLAEngine] Loaded {loaded} weights from {model_path}")

    model = model.to(device=device, dtype=dtype).eval()
    return model, model_path


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class FLAEngine:
    """Single-rank inference engine for FLA recurrent models.

    Public methods mirror ``LlamaEngine``:
      - ``generate(prompts, sampling_params, use_tqdm=False, collect_logits=False)``

    Internal scheduling is much simpler than ``LlamaEngine``:

      1. **Prefill phase** (one seq at a time, full prompt in a single
         forward) initialises each seq's per-layer recurrent state and
         samples the first decode token. Doing prefills sequentially
         (rather than padded-batched) is the simplest correct option and
         matches FLA's reference behaviour for varlen prompts.

      2. **Decode phase** batches across all currently-running seqs:
         each step gathers per-seq state into a single ``[B, H, K, V]``
         tensor per layer, runs one ``[B, 1, D]`` forward, then scatters
         the updated state back. Newly-finished seqs are dropped from the
         batch on the next step.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        max_num_seqs: int = 256,
        trust_remote_code: bool = True,
    ):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.seed = seed
        self.max_num_seqs = max_num_seqs
        self.device = torch.device(device)
        self.dtype = dtype
        self._set_seeds(seed)

        self.model, self.model_path = _load_model(model_name, dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Cache the ordered list of L2 attention module ids so state
        # gather/scatter walks them in deterministic, layer-natural order.
        from kb_nano.tasks.baseline.L2.gla_attention import GatedLinearAttention
        from kb_nano.tasks.baseline.L2.rwkv7_attention import RWKV7Attention
        attn_classes = (GatedLinearAttention, RWKV7Attention)
        self._attn_layer_ids: list[int] = [
            id(m) for m in self.model.modules() if isinstance(m, attn_classes)
        ]
        self._is_rwkv7 = any(
            isinstance(m, RWKV7Attention) for m in self.model.modules()
        )

    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -- sampling -------------------------------------------------------
    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        """Sample one token id from a ``[V]`` logits row."""
        if params.temperature == 0.0:
            return int(logits.argmax(dim=-1).item())
        scaled = logits.float() / params.temperature
        if params.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum - probs >= params.top_p
            sorted_logits[mask] = float("-inf")
            scaled = scaled.scatter(0, sorted_idx, sorted_logits)
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    # -- state plumbing -------------------------------------------------
    def _build_batched_cache(self, active: list[_ActiveSeq]):
        """Stack per-seq states into a batched RecurrentCache.

        Each layer's state tensors must have the same shape across
        sequences (model architecture guarantees this). Sequences that
        haven't seen a forward yet contribute zero-init state.
        """
        from kb_nano.tasks.baseline.L4.recurrent_cache import RecurrentCache

        cache = RecurrentCache()
        for layer_id in self._attn_layer_ids:
            ref = next(
                (s.states[layer_id] for s in active if layer_id in s.states),
                None,
            )
            if ref is None:
                continue  # no seq has run a forward yet for this layer
            shape = ref.shape  # [1, H, K, V]
            stacked = torch.empty(
                (len(active),) + shape[1:], dtype=ref.dtype, device=ref.device,
            )
            for i, seq in enumerate(active):
                if layer_id in seq.states:
                    stacked[i] = seq.states[layer_id][0]
                else:
                    stacked[i].zero_()
            cache.states[layer_id] = stacked
        return cache

    def _scatter_batched_cache(
        self, active: list[_ActiveSeq], cache,
    ) -> None:
        """Write the post-forward batched state back to per-seq slots."""
        for layer_id, batched in cache.states.items():
            for i, seq in enumerate(active):
                seq.states[layer_id] = batched[i:i + 1].detach()

    # -- prefill / decode primitives -----------------------------------
    @torch.no_grad()
    def _prefill_one(self, seq: _ActiveSeq) -> int:
        """Run prefill for ``seq``, store its per-layer state, and
        sample the first decode token."""
        from kb_nano.tasks.baseline.L4.recurrent_cache import RecurrentCache

        ids = torch.tensor([seq.prompt_ids], dtype=torch.long, device=self.device)
        cache = RecurrentCache()
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        for layer_id, state in out.past_key_values.states.items():
            seq.states[layer_id] = state.detach()
        next_id = self._sample(out.logits[0, -1], seq.sampling)
        return next_id

    @torch.no_grad()
    def _decode_step(self, active: list[_ActiveSeq]) -> list[int]:
        """One batched decode step over the active seqs.

        Returns the list of newly-sampled token ids in the same order as
        ``active``.
        """
        ids = torch.tensor(
            [[seq.generated_ids[-1] if seq.generated_ids else seq.prompt_ids[-1]]
             for seq in active],
            dtype=torch.long, device=self.device,
        )
        cache = self._build_batched_cache(active)
        out = self.model(
            input_ids=ids, past_key_values=cache, use_cache=True,
        )
        self._scatter_batched_cache(active, out.past_key_values)
        last_logits = out.logits[:, -1, :]
        return [self._sample(last_logits[i], seq.sampling)
                for i, seq in enumerate(active)]

    # -- public API ----------------------------------------------------
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        collect_logits: bool = False,
        use_tqdm: bool = False,
    ) -> list[GenerationOutput]:
        """Greedy / top-p generate for a batch of prompts.

        ``prompts`` may be raw strings (will be tokenized) or already-
        tokenised id lists. Output order matches ``prompts`` order.
        """
        if collect_logits:
            raise NotImplementedError("collect_logits is not yet supported by FLAEngine.")

        sp_list = (sampling_params if isinstance(sampling_params, list)
                   else [sampling_params] * len(prompts))
        seed = sp_list[0].seed
        if seed is not None:
            self._set_seeds(seed)

        eos = self.tokenizer.eos_token_id
        all_seqs: list[_ActiveSeq] = []
        for i, p in enumerate(prompts):
            ids = p if isinstance(p, list) else self.tokenizer.encode(p)
            sp = sp_list[i]
            all_seqs.append(_ActiveSeq(
                seq_id=i,
                prompt_ids=list(ids),
                max_tokens=sp.max_tokens,
                ignore_eos=sp.ignore_eos,
                sampling=sp,
            ))

        pbar = None
        if use_tqdm:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=len(all_seqs), desc="FLAEngine prompts")

        # Phase 1 — sequential prefill.
        # Done one-at-a-time so each seq gets its own clean cache.
        # Batched-prefill with left padding could be added later but
        # complicates state init for left-padded slots.
        waiting: deque[_ActiveSeq] = deque(all_seqs)
        running: list[_ActiveSeq] = []
        while waiting and len(running) < self.max_num_seqs:
            seq = waiting.popleft()
            first_tok = self._prefill_one(seq)
            seq.generated_ids.append(first_tok)
            if not seq.ignore_eos and eos is not None and first_tok == eos:
                seq.finished = True
            elif len(seq.generated_ids) >= seq.max_tokens:
                seq.finished = True
            running.append(seq)

        # Phase 2 — batched decode loop.
        while running:
            active = [s for s in running if not s.finished]
            if not active:
                break
            tokens = self._decode_step(active)
            for seq, tok in zip(active, tokens):
                seq.generated_ids.append(tok)
                if not seq.ignore_eos and eos is not None and tok == eos:
                    seq.finished = True
                elif len(seq.generated_ids) >= seq.max_tokens:
                    seq.finished = True
            # Drop fully-finished seqs from the running list to admit
            # any waiting seqs (rare path: only triggers if max_num_seqs
            # was hit during prefill).
            still_running = [s for s in running if not s.finished]
            while waiting and len(still_running) < self.max_num_seqs:
                seq = waiting.popleft()
                first_tok = self._prefill_one(seq)
                seq.generated_ids.append(first_tok)
                if not seq.ignore_eos and eos is not None and first_tok == eos:
                    seq.finished = True
                elif len(seq.generated_ids) >= seq.max_tokens:
                    seq.finished = True
                still_running.append(seq)
                if pbar is not None:
                    pbar.update(0)
            running = still_running
            if pbar is not None:
                # update once per fully-finished seq
                done_now = sum(1 for s in all_seqs if s.finished and getattr(s, "_pbar_counted", False) is False)
                for s in all_seqs:
                    if s.finished and not getattr(s, "_pbar_counted", False):
                        s._pbar_counted = True
                if done_now:
                    pbar.update(done_now)

        if pbar is not None:
            pbar.close()

        outputs: list[GenerationOutput] = []
        for seq in all_seqs:
            text = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
            prompt_text = (seq.prompt_ids if isinstance(prompts[seq.seq_id], list)
                           else prompts[seq.seq_id])
            outputs.append(GenerationOutput(
                prompt=prompt_text if isinstance(prompt_text, str) else "",
                generated_text=text,
                token_ids=list(seq.generated_ids),
            ))
        return outputs
