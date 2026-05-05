"""Jamba inference engine.

Distinct from :class:`infra.engine.LlamaEngine` and
:class:`infra.fla_engine.FLAEngine` because Jamba is a *hybrid* model
that needs both kinds of cache simultaneously:

  * 4 (out of 32 in v0.1) attention layers with a transformer-style KV
    cache (``[B, num_kv_heads, T, head_dim]``).
  * 28 Mamba layers with per-sequence selective-scan state
    (``conv_state``: ``[num_slots, intermediate, K-1]``;
    ``ssm_state``:  ``[num_slots, intermediate, ssm_state_size]``).

The full vLLM v1 hybrid scheduler does paged KV + slot-allocated mamba
state with chunked prefill and CUDA graph capture.  That's a lot of
machinery.  For this engine we stay closer to FLAEngine's design:
single-rank, left-padded HF-style batched ``.generate``-equivalent
with a pre-allocated KV slab and live mamba state slabs.

Layout: every batch is ``[B, T]`` with left padding.  Both prefill and
decode use the *same* flat-varlen layout into the Mamba kernels
(query_start_loc, cache_indices) so we reuse vLLM's SOTA Mamba
kernels directly.

Tensor parallel: NOT supported -- single-GPU only.  Open Jamba models
fit on a B200 (Jamba-tiny-dev = 318M, Jamba-v0.1 = 52B).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

# Re-export the same SamplingParams / GenerationOutput dataclasses the
# rest of the codebase uses.  We import them lazily because
# ``infra.engine`` transitively imports the entire model zoo (including
# vision pipelines that need flash-attn), and JambaEngine has no need
# for any of that.  The fallback definitions match the originals
# field-for-field so callers can use either dataclass interchangeably.
try:
    from .engine import GenerationOutput, SamplingParams
except ImportError:  # pragma: no cover -- minimal-deps fallback
    from dataclasses import dataclass as _dc
    from dataclasses import field as _field

    @_dc
    class SamplingParams:  # type: ignore[no-redef]
        temperature: float = 0.0
        top_p: float = 1.0
        max_tokens: int = 512
        seed: int | None = None
        ignore_eos: bool = False

    @_dc
    class GenerationOutput:  # type: ignore[no-redef]
        prompt: str
        generated_text: str
        token_ids: list[int] = _field(default_factory=list)
        logits_history: list | None = None

__all__ = ["JambaEngine", "SamplingParams", "GenerationOutput"]


# ---------------------------------------------------------------------------
# Decode-step CUDA graph entry.  We capture exactly one graph at
# ``B = max_num_seqs`` and pad smaller batches up to that size, the same
# way vLLM captures graphs for its largest configured batch size.
# Static buffers are mutated in-place between replays; only the host
# tensors that drive the loop need to be updated:
#
#   step_input_ids : [B, 1]    int64  -- previous step's sampled tokens
#   slot_pos       : [1]       int64  -- position in KV slab to write
#   decode_mask    : [B,1,1,T] bf16   -- additive mask (we unmask one
#                                        new position per step in-place)
#   next_tokens    : [B]       int64  -- argmax output (read back to host)
# ---------------------------------------------------------------------------
@dataclass
class _JambaDecodeGraph:
    graph: torch.cuda.CUDAGraph
    step_input_ids: torch.Tensor
    slot_pos: torch.Tensor
    decode_mask: torch.Tensor
    next_tokens: torch.Tensor


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class JambaEngine:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        max_num_seqs: int = 64,
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        graph_max_total: int = 2048,
    ):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from ..tasks.baseline.L4.jamba import JambaConfig, JambaForCausalLM

        self.model_name = model_name
        self.seed = seed
        self.max_num_seqs = max_num_seqs
        self.device = torch.device(device)
        self.dtype = dtype
        self._set_seeds(seed)

        model_path = snapshot_download(
            model_name, allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        )
        self.model_path = model_path

        self.config = JambaConfig.from_pretrained(model_path)
        self.config.dtype = dtype
        self.max_model_len = max_model_len  # caller decides

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Build the model on CPU then move to GPU + cast to dtype.
        print(f"  [JambaEngine] Loading config from {model_path}")
        print(f"  [JambaEngine] Architecture: "
              f"L={self.config.num_hidden_layers}, "
              f"hidden={self.config.hidden_size}, "
              f"attn_layers={self.config.attn_layer_offset}/"
              f"{self.config.attn_layer_period}, "
              f"moe_layers={self.config.expert_layer_offset}/"
              f"{self.config.expert_layer_period}, "
              f"experts={self.config.num_experts}, "
              f"top_k={self.config.num_experts_per_tok}")

        with torch.device("cpu"):
            self.model = JambaForCausalLM(self.config)

        n_loaded = self.model.load_weights(model_path)
        print(f"  [JambaEngine] Loaded {n_loaded} weight tensors")

        # Move to GPU. Be careful: A and conv_state must remain fp32 /
        # bf16 as configured -- cast everything to ``dtype`` for now,
        # then explicitly restore A to fp32 (the Mamba SSM kernels
        # require ``A`` to be float32).
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        for layer in self.model.model.layers:
            mamba = getattr(layer, "mamba", None)
            if mamba is not None:
                mamba.A.data = mamba.A.data.float()

        torch.cuda.synchronize()

        # Cache shapes (per layer) used to allocate per-batch buffers.
        cfg = self.config
        self._mamba_intermediate = cfg.mamba_expand * cfg.hidden_size
        self._mamba_d_state = cfg.mamba_d_state
        self._mamba_conv_kernel = cfg.mamba_d_conv
        self._n_mamba_layers = len(self.model.model.mamba_layer_indices)
        self._n_attn_layers = len(self.model.model.attention_layer_indices)
        self._head_dim = cfg.hidden_size // cfg.num_attention_heads
        self._n_kv_heads = cfg.num_key_value_heads

        # CUDA-graph capture cache for the decode step.  All graphs
        # share a fixed ``graph_max_total`` capacity so the static
        # buffers (KV slabs, mask) have stable shape; varying live
        # decode lengths are handled by the mask rather than by
        # re-capturing a new graph per shape.  vLLM uses the same trick
        # (paged KV makes the attention shape constant).
        #
        # Single-bucket strategy: we capture exactly one decode graph
        # at ``B = max_num_seqs`` and pad smaller batches up to that
        # size.  Pre-capturing once at init time avoids the well-known
        # "graph pool overlaps caching-allocator block" issue that
        # can corrupt replay state when a graph is captured *after*
        # other work has fragmented the pool.
        self.graph_max_total = graph_max_total
        self._decode_graph: _JambaDecodeGraph | None = None
        self._decode_graph_buffers: dict | None = None
        # Disable graph capture if requested (e.g. for debugging,
        # profiling the eager path, or when the CUDA driver disallows
        # graph capture in the current process).
        self._use_cuda_graphs = (
            os.environ.get("KB_NANO_JAMBA_CUDA_GRAPHS", "1") not in ("0", "false", "False")
        )

        # Pre-capture the decode graph immediately so the graph's
        # private memory pool is reserved BEFORE the caching allocator
        # has a chance to fragment GPU memory with prefill / mask
        # tensors on subsequent ``_run_batch`` calls.  Skipping this
        # leads to ``cudaErrorIllegalAddress`` on graph replay when
        # the allocator hands out a block from inside the captured
        # graph's reserved region.
        if self._use_cuda_graphs:
            self._capture_decode_graph()

    # ------------------------------------------------------------------
    # Random seeds
    # ------------------------------------------------------------------
    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Static (per-shape) decode buffers.  Allocated lazily on first use
    # and reused across ``_run_batch`` calls so the captured CUDA graph's
    # tensor pointers stay valid.  The contents are reset at the start
    # of each batch.
    # ------------------------------------------------------------------
    def _get_or_alloc_static_buffers(self) -> dict:
        if self._decode_graph_buffers is not None:
            return self._decode_graph_buffers

        device = self.device
        batch_size = self.max_num_seqs
        max_total = self.graph_max_total
        # KV slabs (per attention layer): [B, H_kv, max_total, D].
        kv_slabs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self._n_attn_layers):
            k = torch.zeros(
                batch_size, self._n_kv_heads, max_total, self._head_dim,
                dtype=self.dtype, device=device,
            )
            v = torch.zeros(
                batch_size, self._n_kv_heads, max_total, self._head_dim,
                dtype=self.dtype, device=device,
            )
            kv_slabs.append((k, v))

        # Mamba state slabs (per mamba layer).
        conv_states: list[torch.Tensor] = []
        ssm_states: list[torch.Tensor] = []
        K_minus_1 = max(self._mamba_conv_kernel - 1, 1)
        for _ in range(self._n_mamba_layers):
            raw_conv = torch.zeros(
                batch_size, K_minus_1, self._mamba_intermediate,
                dtype=self.dtype, device=device,
            )
            conv_states.append(raw_conv.transpose(-1, -2))
            ssm_states.append(torch.zeros(
                batch_size, self._mamba_intermediate, self._mamba_d_state,
                dtype=self.dtype, device=device,
            ))

        cache_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        # Decode-step inputs / outputs (shapes fixed at graph capture).
        step_input_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device,
        )
        slot_pos = torch.zeros(1, dtype=torch.long, device=device)
        # Mask is initialised in ``_run_batch`` per-batch (because it
        # depends on the prompt-padding pattern).  We allocate it here
        # so its identity is stable for graph capture.
        decode_mask = torch.zeros(
            batch_size, 1, 1, max_total, dtype=self.dtype, device=device,
        )
        next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

        bufs = {
            "kv_slabs": kv_slabs,
            "conv_states": conv_states,
            "ssm_states": ssm_states,
            "cache_indices": cache_indices,
            "step_input_ids": step_input_ids,
            "slot_pos": slot_pos,
            "decode_mask": decode_mask,
            "next_tokens": next_tokens,
        }
        self._decode_graph_buffers = bufs
        return bufs

    def _capture_decode_graph(self) -> _JambaDecodeGraph:
        """Capture (or replay-cached) the decode-step CUDA graph.

        The graph reads from the static buffers in ``_decode_graph_buffers``
        and writes the sampled token to ``next_tokens``.  The host loop
        in ``_run_batch`` must:

          1. Update ``step_input_ids`` (in-place) to the previous step's
             output token (or the prefill output for the first step).
          2. Update ``slot_pos[0]`` to ``cur_len - 1``.
          3. Unmask position ``cur_len - 1`` in ``decode_mask`` (set to 0).
          4. Replay the graph.
          5. (Async) read ``next_tokens`` into a host history buffer.
        """
        if self._decode_graph is not None:
            return self._decode_graph

        bufs = self._get_or_alloc_static_buffers()

        # Closure: one decode step, fully static-shape.
        def _decode_step():
            hidden, _ = self.model(
                input_ids=bufs["step_input_ids"],
                attn_kv_slabs=bufs["kv_slabs"],
                attn_slot_pos=bufs["slot_pos"],
                attn_mask_4d=bufs["decode_mask"],
                mamba_conv_states=bufs["conv_states"],
                mamba_ssm_states=bufs["ssm_states"],
                mamba_cache_indices=bufs["cache_indices"],
                mamba_query_start_loc=None,
                mamba_has_initial_state=None,
                mamba_is_decode=True,
                mamba_pad_mask_flat=None,
            )
            logits = self.model.compute_logits(hidden[:, -1, :])
            tok = logits.argmax(dim=-1)
            # Write into the persistent buffer.  Must be in-place so the
            # graph's output tensor identity is fixed.
            bufs["next_tokens"].copy_(tok)

        # Warmup outside the graph-capture stream to populate workspace
        # tensors / autotune caches.  vLLM does this in ``_dummy_run``.
        # Using a fresh stream that's joined back to the current stream
        # ensures all allocator state is settled before capture begins.
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _decode_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # Drop any cached allocator blocks that might overlap with the
        # graph's private pool.  Without this, a second (B, max_total)
        # capture can hit ``illegal memory access`` on replay because
        # PyTorch's caching allocator may hand out a block from inside
        # the previous graph's reserved pool.
        torch.cuda.empty_cache()

        # Each (batch_size, max_total) gets its OWN private graph pool.
        # Sharing a pool across distinct decode shapes is unsafe: the
        # captured graphs' workspace tensors can alias and produce
        # ``illegal memory access`` on replay.  vLLM solves this with a
        # dedicated ``MemoryPool`` per shape bucket; we mirror that.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _decode_step()

        entry = _JambaDecodeGraph(
            graph=graph,
            step_input_ids=bufs["step_input_ids"],
            slot_pos=bufs["slot_pos"],
            decode_mask=bufs["decode_mask"],
            next_tokens=bufs["next_tokens"],
        )
        self._decode_graph = entry
        return entry

    # ------------------------------------------------------------------
    # Per-batch cache allocation (legacy, used by the eager fallback path)
    # ------------------------------------------------------------------
    def _alloc_kv_slabs(self, batch_size: int, max_len: int):
        """Pre-allocate growable KV slabs, one (k, v) per attention layer.

        Returns a list of length ``self._n_attn_layers``, each a
        ``(k_buf, v_buf)`` pair of shape
        ``[B, num_kv_heads, max_len, head_dim]``.
        """
        slabs = []
        for _ in range(self._n_attn_layers):
            k = torch.zeros(
                batch_size, self._n_kv_heads, max_len, self._head_dim,
                dtype=self.dtype, device=self.device,
            )
            v = torch.zeros(
                batch_size, self._n_kv_heads, max_len, self._head_dim,
                dtype=self.dtype, device=self.device,
            )
            slabs.append((k, v))
        return slabs

    def _alloc_mamba_states(self, batch_size: int):
        """Allocate per-layer Mamba conv & SSM state slabs.

        One slot per active row, indexed via ``cache_indices = arange(B)``.
        Shapes:
          ``conv_state``: kernel-view ``[B, intermediate, K-1]`` with
                          ``stride(intermediate) == 1`` (the kernel
                          asserts this).  We allocate as ``[B, K-1,
                          intermediate]`` and transpose the last two
                          dims to satisfy the contract -- matches
                          how ``MambaStateManager`` does it for the
                          paged-engine mixers.
          ``ssm_state``:  ``[B, intermediate, ssm_state_size]``.  The
                          ``selective_*`` kernels are tolerant to the
                          ``ssm_state`` stride layout, so we allocate
                          contiguous in that order.
        """
        conv_states: list[torch.Tensor] = []
        ssm_states: list[torch.Tensor] = []
        K_minus_1 = max(self._mamba_conv_kernel - 1, 1)
        for _ in range(self._n_mamba_layers):
            # Allocate [B, K-1, intermediate] and transpose to
            # [B, intermediate, K-1].  This makes stride(intermediate) = 1.
            raw_conv = torch.zeros(
                batch_size, K_minus_1, self._mamba_intermediate,
                dtype=self.dtype, device=self.device,
            )
            conv_states.append(raw_conv.transpose(-1, -2))
            ssm_states.append(torch.zeros(
                batch_size, self._mamba_intermediate, self._mamba_d_state,
                dtype=self.dtype, device=self.device,
            ))
        return conv_states, ssm_states

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _greedy_argmax(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def _sample_step(
        self, logits: torch.Tensor, sampling_params: list[SamplingParams],
    ) -> list[int]:
        """logits: ``[B, V]`` -- one sampled token per row."""
        if all(p.temperature == 0.0 for p in sampling_params):
            return self._greedy_argmax(logits).tolist()
        out: list[int] = []
        for i, p in enumerate(sampling_params):
            row = logits[i]
            if p.temperature == 0.0:
                out.append(int(row.argmax().item()))
                continue
            scaled = row.float() / p.temperature
            if p.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                mask = cum - probs >= p.top_p
                sorted_logits[mask] = float("-inf")
                scaled = scaled.scatter(0, sorted_idx, sorted_logits)
            probs = torch.softmax(scaled, dim=-1)
            out.append(int(torch.multinomial(probs, 1).item()))
        return out

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[GenerationOutput]:
        """Batched generate.  Splits ``prompt_token_ids`` into micro-
        batches of size ``self.max_num_seqs`` and runs each through
        the standard prefill -> token-by-token decode loop.
        """
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompt_token_ids)
        assert len(sampling_params) == len(prompt_token_ids)

        outputs: list[GenerationOutput | None] = [None] * len(prompt_token_ids)

        n = len(prompt_token_ids)
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=n, desc="kb-nano Jamba")
        else:
            pbar = None

        for start in range(0, n, self.max_num_seqs):
            end = min(start + self.max_num_seqs, n)
            indices = list(range(start, end))
            batch_prompts = [prompt_token_ids[i] for i in indices]
            batch_sps = [sampling_params[i] for i in indices]
            batch_outputs = self._run_batch(batch_prompts, batch_sps)
            for i, out in zip(indices, batch_outputs):
                outputs[i] = out
            if pbar is not None:
                pbar.update(len(indices))
        if pbar is not None:
            pbar.close()
        return outputs  # type: ignore[return-value]

    @torch.inference_mode()
    def _run_batch(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: list[SamplingParams],
    ) -> list[GenerationOutput]:
        B = len(prompt_token_ids)
        device = self.device
        eos = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        max_out = max(p.max_tokens for p in sampling_params)
        prompt_lens = [len(p) for p in prompt_token_ids]
        max_prompt = max(prompt_lens)
        max_total = max_prompt + max_out

        # CUDA-graph eligibility:
        #   - greedy decoding for every row (graph samples via argmax)
        #   - ignore_eos OR all rows have the same max_tokens budget; we
        #     loop a fixed number of steps and ignore early-stop in the
        #     graph path. To stay safe we trim to per-row max_tokens
        #     after the fact.
        #   - the batch's live ``max_total`` fits in the engine's static
        #     ``graph_max_total`` buffer.  Larger requests fall back to
        #     the eager path.
        all_greedy = all(p.temperature == 0.0 for p in sampling_params)
        fits_static = max_total <= self.graph_max_total
        use_graph = self._use_cuda_graphs and all_greedy and fits_static

        # ------------------------------------------------------------------
        # Build left-padded prompt tensor + attention mask (1 for real
        # tokens, 0 for pad).  HF Jamba uses left padding for batched
        # generation since causal masking aligns at the right.
        # ------------------------------------------------------------------
        input_ids = torch.full((B, max_prompt), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, max_prompt), dtype=torch.long, device=device)
        for i, p in enumerate(prompt_token_ids):
            offset = max_prompt - len(p)
            input_ids[i, offset:] = torch.tensor(p, dtype=torch.long, device=device)
            attention_mask[i, offset:] = 1

        if use_graph:
            return self._run_batch_graph(
                prompt_token_ids, sampling_params,
                B=B, max_prompt=max_prompt, max_out=max_out, max_total=max_total,
                input_ids=input_ids, attention_mask=attention_mask,
                pad_id=pad_id,
            )

        # ==============================================================
        # Eager fallback (non-greedy, or CUDA graphs disabled).
        # ==============================================================
        kv_slabs = self._alloc_kv_slabs(B, max_total)
        conv_states, ssm_states = self._alloc_mamba_states(B)
        cache_indices = torch.arange(B, dtype=torch.int32, device=device)

        attn_mask_4d = self._build_prefill_attn_mask(attention_mask, max_prompt)
        attn_writeback = [
            (kv[0][:, :, :max_prompt, :], kv[1][:, :, :max_prompt, :])
            for kv in kv_slabs
        ]
        mamba_qsl_p = torch.tensor(
            [i * max_prompt for i in range(B + 1)],
            dtype=torch.int32, device=device,
        )
        mamba_has_init = torch.zeros(B, dtype=torch.bool, device=device)
        mamba_pad_flat = attention_mask.bool().reshape(-1)

        hidden, _ = self.model(
            input_ids=input_ids,
            attn_past_kv=None,
            attn_cache_writeback=attn_writeback,
            attn_mask_4d=attn_mask_4d,
            mamba_conv_states=conv_states,
            mamba_ssm_states=ssm_states,
            mamba_cache_indices=cache_indices,
            mamba_query_start_loc=mamba_qsl_p,
            mamba_has_initial_state=mamba_has_init,
            mamba_is_decode=False,
            mamba_pad_mask_flat=mamba_pad_flat,
        )

        logits = self.model.compute_logits(hidden[:, -1, :])
        next_tokens = self._sample_step(logits, sampling_params)

        generated: list[list[int]] = [[] for _ in range(B)]
        finished = [False] * B
        for i, t in enumerate(next_tokens):
            generated[i].append(int(t))
            if not sampling_params[i].ignore_eos and eos is not None and t == eos:
                finished[i] = True
            if len(generated[i]) >= sampling_params[i].max_tokens:
                finished[i] = True

        cur_len = max_prompt + 1
        decode_key_mask_full = torch.zeros(
            B, 1, 1, max_total, dtype=self.dtype, device=device,
        )
        pad_positions = (attention_mask == 0)
        if pad_positions.any():
            decode_key_mask_full[:, 0, 0, :max_prompt] = torch.where(
                pad_positions,
                torch.full((), torch.finfo(self.dtype).min, device=device, dtype=self.dtype),
                torch.zeros((), device=device, dtype=self.dtype),
            )

        while cur_len <= max_total and not all(finished):
            step_ids = torch.tensor(
                [[generated[i][-1]] for i in range(B)],
                dtype=torch.long, device=device,
            )
            past_kv = [
                (kv[0][:, :, :cur_len - 1, :], kv[1][:, :, :cur_len - 1, :])
                for kv in kv_slabs
            ]
            writeback = [
                (kv[0][:, :, :cur_len, :], kv[1][:, :, :cur_len, :])
                for kv in kv_slabs
            ]
            decode_mask = decode_key_mask_full[:, :, :, :cur_len]

            hidden, _ = self.model(
                input_ids=step_ids,
                attn_past_kv=past_kv,
                attn_cache_writeback=writeback,
                attn_mask_4d=decode_mask,
                mamba_conv_states=conv_states,
                mamba_ssm_states=ssm_states,
                mamba_cache_indices=cache_indices,
                mamba_query_start_loc=None,
                mamba_has_initial_state=None,
                mamba_is_decode=True,
                mamba_pad_mask_flat=None,
            )
            logits = self.model.compute_logits(hidden[:, -1, :])
            next_tokens = self._sample_step(logits, sampling_params)
            for i, t in enumerate(next_tokens):
                if finished[i]:
                    continue
                generated[i].append(int(t))
                if not sampling_params[i].ignore_eos and eos is not None and t == eos:
                    finished[i] = True
                if len(generated[i]) >= sampling_params[i].max_tokens:
                    finished[i] = True
            cur_len += 1

        return self._materialise(generated)

    # ------------------------------------------------------------------
    # CUDA-graph fast path: greedy decode using a captured single-step
    # graph + in-place buffer updates.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _run_batch_graph(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: list[SamplingParams],
        *,
        B: int,
        max_prompt: int,
        max_out: int,
        max_total: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_id: int,
    ) -> list[GenerationOutput]:
        """Greedy-decode using a captured CUDA graph at fixed
        B = ``max_num_seqs``.  Smaller ``B`` is padded up to
        ``max_num_seqs`` using a dummy prompt; the dummy outputs are
        discarded.  This single-bucket strategy (vs per-B captures)
        avoids the well-known issue where a captured graph's reserved
        memory pool can be reused by the caching allocator after
        capture, producing illegal-memory-access on replay.
        """
        device = self.device
        eos = self.tokenizer.eos_token_id
        B_pad = self.max_num_seqs

        # ---- Pad prompts up to B_pad with a dummy sequence ----
        if B < B_pad:
            # Pad with copies of the first prompt's input_ids.  The
            # padding rows are computed but their outputs are dropped.
            input_ids = self._pad_to_b(input_ids, B_pad, pad_id)
            attention_mask = self._pad_to_b(attention_mask, B_pad, 0)

        # ---- Capture graph (no-op after first call) ----
        graph_entry = self._capture_decode_graph()

        # ---- Get static buffers ----
        bufs = self._get_or_alloc_static_buffers()
        kv_slabs = bufs["kv_slabs"]
        conv_states = bufs["conv_states"]
        ssm_states = bufs["ssm_states"]
        cache_indices = bufs["cache_indices"]
        step_input_ids = bufs["step_input_ids"]
        slot_pos = bufs["slot_pos"]
        decode_mask = bufs["decode_mask"]
        next_tokens_buf = bufs["next_tokens"]

        # Reset state to zero so prefill starts from a clean slate.
        for k, v in kv_slabs:
            k.zero_()
            v.zero_()
        for cs in conv_states:
            cs.zero_()
        for ss in ssm_states:
            ss.zero_()

        # ---- Prefill (eager -- one-shot per batch, no graph) ----
        attn_writeback = [
            (kv[0][:, :, :max_prompt, :], kv[1][:, :, :max_prompt, :])
            for kv in kv_slabs
        ]
        attn_mask_4d_p = self._build_prefill_attn_mask(attention_mask, max_prompt)

        mamba_qsl_p = torch.tensor(
            [i * max_prompt for i in range(B_pad + 1)],
            dtype=torch.int32, device=device,
        )
        mamba_has_init = torch.zeros(B_pad, dtype=torch.bool, device=device)
        mamba_pad_flat = attention_mask.bool().reshape(-1)

        hidden, _ = self.model(
            input_ids=input_ids,
            attn_past_kv=None,
            attn_cache_writeback=attn_writeback,
            attn_mask_4d=attn_mask_4d_p,
            mamba_conv_states=conv_states,
            mamba_ssm_states=ssm_states,
            mamba_cache_indices=cache_indices,
            mamba_query_start_loc=mamba_qsl_p,
            mamba_has_initial_state=mamba_has_init,
            mamba_is_decode=False,
            mamba_pad_mask_flat=mamba_pad_flat,
        )
        prefill_logits = self.model.compute_logits(hidden[:, -1, :])
        first_tok = prefill_logits.argmax(dim=-1)  # [B_pad]

        # ---- Initialise the static decode mask ----
        # Layout: [B_pad, 1, 1, graph_max_total].  All positions
        # ``>= max_prompt`` start masked.  Padded prompt positions
        # are also masked.  We unmask one new position per step.
        mask_min = torch.finfo(self.dtype).min
        decode_mask.fill_(mask_min)
        valid_prompt = attention_mask.bool()  # [B_pad, max_prompt]
        decode_mask[:, 0, 0, :max_prompt].masked_fill_(valid_prompt, 0.0)

        # ---- Decode loop driven by graph replay ----
        per_row_max = [p.max_tokens for p in sampling_params]
        global_max = max_out

        tok_history = torch.empty(
            global_max, B_pad, dtype=torch.long, pin_memory=True,
        )
        tok_history[0].copy_(first_tok, non_blocking=True)
        step_input_ids[:, 0].copy_(first_tok)

        cur_len = max_prompt  # position the FIRST decode token writes to
        for step in range(1, global_max):
            slot_pos.fill_(cur_len)
            decode_mask[:, 0, 0, cur_len].fill_(0.0)
            graph_entry.graph.replay()
            tok_history[step].copy_(next_tokens_buf, non_blocking=True)
            step_input_ids[:, 0].copy_(next_tokens_buf)
            cur_len += 1

        torch.cuda.synchronize()

        # ---- Build per-row generated lists (drop padding rows) ----
        history_t = tok_history.numpy()
        generated: list[list[int]] = [[] for _ in range(B)]
        for i in range(B):
            limit = per_row_max[i]
            tokens_i: list[int] = []
            for s in range(min(global_max, limit)):
                t = int(history_t[s, i])
                tokens_i.append(t)
                if (not sampling_params[i].ignore_eos
                        and eos is not None and t == eos):
                    break
            generated[i] = tokens_i

        return self._materialise(generated)

    @staticmethod
    def _pad_to_b(t: torch.Tensor, B_pad: int, pad_value) -> torch.Tensor:
        """Pad a [B, T] tensor to [B_pad, T] with ``pad_value`` rows."""
        b, *rest = t.shape
        if b >= B_pad:
            return t
        extra = B_pad - b
        pad_block = torch.full(
            (extra, *rest), pad_value, dtype=t.dtype, device=t.device,
        )
        return torch.cat([t, pad_block], dim=0)

    def _materialise(self, generated: list[list[int]]) -> list[GenerationOutput]:
        results: list[GenerationOutput] = []
        for tokens in generated:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(GenerationOutput(
                prompt="",
                generated_text=text,
                token_ids=tokens,
            ))
        return results

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------
    def _build_prefill_attn_mask(
        self,
        attention_mask_2d: torch.Tensor,  # [B, T] (1=real, 0=pad)
        T: int,
    ) -> torch.Tensor:
        """Build a 4D additive attention mask for the prefill forward.

        The mask is causal AND incorporates the input-side padding
        (left-pad keys are masked out).  Shape ``[B, 1, T, T]``.

        The masked-out positions use ``finfo(self.dtype).min`` rather
        than ``finfo(float32).min``: PyTorch's SDPA / cuDNN-flash
        kernels promote the mask to the Q/K dtype internally, and an
        ``-3.4e38`` fp32 sentinel becomes ``-inf`` in bf16, which then
        produces ``NaN`` after softmax for any row that is fully masked
        (e.g. a padded query position).  Using the dtype's own
        finfo.min preserves the "very negative but finite" property
        across the cast.
        """
        device = attention_mask_2d.device
        causal = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=device)
        )  # [T, T]
        key_mask = attention_mask_2d.bool().unsqueeze(1)  # [B, 1, T] valid keys
        attn_bool = causal.unsqueeze(0) & key_mask  # [B, T, T]
        mask_min = torch.finfo(self.dtype).min
        attn_add = torch.where(
            attn_bool,
            torch.zeros((), device=device, dtype=self.dtype),
            torch.full((), mask_min, device=device, dtype=self.dtype),
        )
        return attn_add.unsqueeze(1)  # [B, 1, T, T]
