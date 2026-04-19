"""FLA inference engine for recurrent linear-attention models.

Distinct from ``LlamaEngine`` because recurrent models (GLA / RetNet /
RWKV7) carry per-sequence state matrices instead of a paged KV cache.
The semantic gap is too wide to share scheduling code:

  - No paged KV cache: state is a per-layer ``[B, H, K, V]`` tensor.
  - No flat token layout: prefill and decode are batched ``[B, T, D]``
    forwards; padding and per-seq attention masks instead of varlen.
  - No CUDA-graph capture against fixed slot tables: the state buffers
    move whenever the active batch shape changes.

Surface area mirrors ``LlamaEngine`` for the bits ``bench_*.py`` and
user code touch:

  - ``FLAEngine(model_name, dtype, seed, max_num_seqs, ...).generate(prompts, sampling_params)``
  - ``SamplingParams`` and ``GenerationOutput`` are re-exported from
    ``infra.engine`` so callers can use either engine interchangeably.

Scheduling: continuous batching with chunked prefill. Each loop
iteration runs at most ONE B=1 prefill chunk (fixed
``chunked_prefill_size``, default 256) AND one B=N batched decode step
over all sequences whose prefill is complete. New sequences enter the
decode batch as soon as their final prefill chunk produces a sample,
without waiting for the rest of the prefill backlog.

The chunk size is forced to a multiple of 64 (FLA's chunk-vs-recurrent
threshold) and the *last* chunk of each prompt is absorbed if it would
leave a sub-64-token tail; this keeps every chunk on the chunk kernel
and avoids tiny FP differences vs single-shot prefill that can flip
argmax for low-confidence tokens.

Tensor parallel is not implemented (TP=1 only) — all FLA models we
target are < 10B params and fit on a single H200.
"""

from __future__ import annotations

import os
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from glob import glob
from typing import Any

import numpy as np
import torch

from .engine import GenerationOutput, SamplingParams  # re-exported

__all__ = ["FLAEngine", "SamplingParams", "GenerationOutput"]


# Suppress one expected FLA warning: ``seq_len < num_heads`` triggers a
# layout heuristic check during T=1 RWKV7 decode. Our shapes are correct.
warnings.filterwarnings(
    "ignore",
    message=r".*seq_len.*<.*num_heads.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Per-sequence bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class _ActiveSeq:
    """Bookkeeping for a single in-flight sequence."""
    seq_id: int
    prompt_ids: list[int]
    sampling: SamplingParams
    max_tokens: int = 512
    ignore_eos: bool = False

    # prefill progress (number of prompt tokens already consumed)
    prefill_pos: int = 0
    # tokens we've sampled (does not include the prompt)
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False

    # per-layer recurrent / conv state (each shape [1, ...])
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
    """Load an FLA model via snapshot_download + safetensors weight copy."""
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
    """

    # FLA's chunk-kernel boundary. Sub-64 tails fall back to fused-recurrent
    # which is mathematically equivalent but not bit-identical, so we never
    # leave a chunk smaller than this.
    _CHUNK_BOUNDARY = 64

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        max_num_seqs: int = 256,
        chunked_prefill_size: int = 256,
        trust_remote_code: bool = True,
    ):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.seed = seed
        self.max_num_seqs = max_num_seqs
        # Round up to a multiple of 64 so every prefill chunk dispatches
        # to the chunk kernel (not fused-recurrent), matching single-shot
        # prefill bit-for-bit at the chunk boundary.
        cps = max(chunked_prefill_size, self._CHUNK_BOUNDARY)
        self.chunked_prefill_size = ((cps + self._CHUNK_BOUNDARY - 1)
                                     // self._CHUNK_BOUNDARY) * self._CHUNK_BOUNDARY
        self.device = torch.device(device)
        self.dtype = dtype
        self._set_seeds(seed)

        self.model, self.model_path = _load_model(model_name, dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Cache the ordered list of L2 attention + FFN module ids so state
        # gather/scatter walks them in deterministic, layer-natural order.
        from kb_nano.tasks.baseline.L2.gla_attention import GatedLinearAttention
        from kb_nano.tasks.baseline.L2.rwkv7_attention import RWKV7Attention
        from kb_nano.tasks.baseline.L2.rwkv7_ffn import RWKV7FeedForward

        attn_classes = (GatedLinearAttention, RWKV7Attention)
        # Order matters for stable cache walks
        self._attn_layer_ids: list[int] = [
            id(m) for m in self.model.modules() if isinstance(m, attn_classes)
        ]
        # Conv-state-bearing modules (RWKV7 attn + FFN). Order matches
        # discovery order so gather/scatter is deterministic.
        self._conv_module_ids: list[int] = [
            id(m) for m in self.model.modules()
            if isinstance(m, (RWKV7Attention, RWKV7FeedForward))
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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
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

    def _sample_batch(
        self, logits: torch.Tensor, seqs: list[_ActiveSeq],
    ) -> list[int]:
        # Greedy fast path (most common)
        if all(s.sampling.temperature == 0.0 for s in seqs):
            return logits.argmax(dim=-1).tolist()
        return [self._sample(logits[i], s.sampling) for i, s in enumerate(seqs)]

    # ------------------------------------------------------------------
    # State plumbing
    # ------------------------------------------------------------------
    def _build_batched_cache(self, active: list[_ActiveSeq], decode: bool):
        """Stack per-seq states into a batched RecurrentCache.

        ``decode=True`` also fills ``seq_offsets`` from each seq's
        total prefix length so RoPE / position-aware kernels see the
        right global token index.
        """
        from kb_nano.tasks.baseline.L4.recurrent_cache import RecurrentCache

        cache = RecurrentCache()
        for layer_id in self._attn_layer_ids:
            ref = next(
                (s.states[layer_id] for s in active if layer_id in s.states),
                None,
            )
            if ref is None:
                continue
            stacked = torch.empty(
                (len(active),) + ref.shape[1:],
                dtype=ref.dtype, device=ref.device,
            )
            for i, seq in enumerate(active):
                if layer_id in seq.states:
                    stacked[i] = seq.states[layer_id][0]
                else:
                    stacked[i].zero_()
            cache.states[layer_id] = stacked

        for mod_id in self._conv_module_ids:
            ref = next(
                (s.conv_states[mod_id] for s in active if mod_id in s.conv_states),
                None,
            )
            if ref is None:
                continue
            stacked = torch.empty(
                (len(active),) + ref.shape[1:],
                dtype=ref.dtype, device=ref.device,
            )
            for i, seq in enumerate(active):
                if mod_id in seq.conv_states:
                    stacked[i] = seq.conv_states[mod_id][0]
                else:
                    stacked[i].zero_()
            cache.conv_states[mod_id] = stacked

        # seq_offsets per row (prefill_pos = number of tokens already in state)
        cache.seq_offsets = torch.tensor(
            [s.prefill_pos for s in active],
            dtype=torch.int64, device=self.device,
        )
        return cache

    def _scatter_batched_cache(
        self, active: list[_ActiveSeq], cache,
    ) -> None:
        # IMPORTANT: clone each slice so the per-seq state owns its own
        # storage. A bare ``batched[i:i+1]`` is a view into the batched
        # tensor; the *entire* batched tensor would then stay alive as
        # long as any seq references its slice, causing a memory leak
        # that grows linearly with the number of decode steps.
        for layer_id, batched in cache.states.items():
            for i, seq in enumerate(active):
                seq.states[layer_id] = batched[i:i + 1].detach().clone()
        for mod_id, batched in cache.conv_states.items():
            for i, seq in enumerate(active):
                seq.conv_states[mod_id] = batched[i:i + 1].detach().clone()

    def _store_single_cache(self, seq: _ActiveSeq, cache) -> None:
        for layer_id, t in cache.states.items():
            seq.states[layer_id] = t.detach().clone()
        for mod_id, t in cache.conv_states.items():
            seq.conv_states[mod_id] = t.detach().clone()

    @staticmethod
    def _drop_seq_state(seq: _ActiveSeq) -> None:
        seq.states.clear()
        seq.conv_states.clear()

    # ------------------------------------------------------------------
    # Prefill / decode primitives
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _prefill_chunk(
        self, seq: _ActiveSeq, chunk_ids: list[int],
    ) -> torch.Tensor | None:
        """Run a single prefill chunk for ``seq``, threading state in/out.

        Returns the per-token logits tensor [1, T, V] only when this
        chunk is the LAST chunk of the prompt (so the engine can sample
        the first generated token). Otherwise returns ``None`` to save
        the upcast / lm_head call.
        """
        from kb_nano.tasks.baseline.L4.recurrent_cache import RecurrentCache

        ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
        cache = RecurrentCache()
        for layer_id, t in seq.states.items():
            cache.states[layer_id] = t
        for mod_id, t in seq.conv_states.items():
            cache.conv_states[mod_id] = t
        cache.seq_offsets = seq.prefill_pos  # int — broadcast across the row

        new_pos = seq.prefill_pos + len(chunk_ids)
        is_last_chunk = new_pos >= len(seq.prompt_ids)

        out = self.model(
            input_ids=ids, past_key_values=cache, use_cache=True,
        )
        self._store_single_cache(seq, out.past_key_values)
        seq.prefill_pos = new_pos
        return out.logits if is_last_chunk else None

    @torch.no_grad()
    def _decode_step(self, active: list[_ActiveSeq]) -> list[int]:
        ids = torch.tensor(
            [[seq.generated_ids[-1]] for seq in active],
            dtype=torch.long, device=self.device,
        )
        cache = self._build_batched_cache(active, decode=True)
        out = self.model(
            input_ids=ids, past_key_values=cache, use_cache=True,
        )
        self._scatter_batched_cache(active, out.past_key_values)
        for seq in active:
            seq.prefill_pos += 1  # decode step appends one token to the state
        return self._sample_batch(out.logits[:, -1, :], active)

    # ------------------------------------------------------------------
    # Chunk planner — tail-absorption to keep every chunk on the chunk
    # kernel (T >= 64) when the prompt allows.
    # ------------------------------------------------------------------
    def _next_chunk_size(self, seq: _ActiveSeq) -> int:
        remaining = len(seq.prompt_ids) - seq.prefill_pos
        if remaining <= self.chunked_prefill_size + self._CHUNK_BOUNDARY:
            # Take everything that's left to avoid leaving a sub-64 tail.
            return remaining
        return self.chunked_prefill_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        collect_logits: bool = False,
        use_tqdm: bool = False,
    ) -> list[GenerationOutput]:
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
                sampling=sp,
                max_tokens=sp.max_tokens,
                ignore_eos=sp.ignore_eos,
            ))

        pbar = None
        if use_tqdm:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=len(all_seqs), desc="FLAEngine prompts")
        finished_count = 0

        waiting: deque[_ActiveSeq] = deque(all_seqs)
        # In-progress prefill (not yet sampled their first token)
        prefilling: list[_ActiveSeq] = []
        # Decoding (have at least one generated token, still under max_tokens)
        running: list[_ActiveSeq] = []

        def _admit() -> None:
            while waiting and (len(prefilling) + len(running)) < self.max_num_seqs:
                prefilling.append(waiting.popleft())

        _admit()

        # Main loop: each iteration runs at most one B=1 prefill chunk and
        # one B=N decode step over running seqs. New seqs that finish
        # prefill are added to ``running`` immediately so they decode on
        # the next iteration.
        while prefilling or running:
            # ---- prefill: pop the head of the prefilling list, run one chunk
            if prefilling:
                seq = prefilling[0]
                chunk_size = self._next_chunk_size(seq)
                chunk = seq.prompt_ids[seq.prefill_pos:seq.prefill_pos + chunk_size]
                logits = self._prefill_chunk(seq, chunk)
                if logits is not None:
                    # Last chunk: sample first generated token, promote.
                    first = self._sample(logits[0, -1], seq.sampling)
                    seq.generated_ids.append(first)
                    if not seq.ignore_eos and eos is not None and first == eos:
                        seq.finished = True
                    elif len(seq.generated_ids) >= seq.max_tokens:
                        seq.finished = True
                    prefilling.pop(0)
                    if seq.finished:
                        finished_count += 1
                        self._drop_seq_state(seq)
                        if pbar is not None:
                            pbar.update(1)
                    else:
                        running.append(seq)
                    _admit()

            # ---- decode: one step over all running seqs
            if running:
                tokens = self._decode_step(running)
                still_running: list[_ActiveSeq] = []
                for seq, tok in zip(running, tokens):
                    seq.generated_ids.append(tok)
                    if not seq.ignore_eos and eos is not None and tok == eos:
                        seq.finished = True
                    elif len(seq.generated_ids) >= seq.max_tokens:
                        seq.finished = True
                    if seq.finished:
                        finished_count += 1
                        self._drop_seq_state(seq)
                        if pbar is not None:
                            pbar.update(1)
                    else:
                        still_running.append(seq)
                running = still_running
                _admit()

        if pbar is not None:
            pbar.close()

        outputs: list[GenerationOutput] = []
        for seq in all_seqs:
            text = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
            prompt_text = prompts[seq.seq_id] if isinstance(prompts[seq.seq_id], str) else ""
            outputs.append(GenerationOutput(
                prompt=prompt_text,
                generated_text=text,
                token_ids=list(seq.generated_ids),
            ))
        return outputs
