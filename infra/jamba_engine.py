"""Jamba inference engine.

Distinct from :class:`infra.engine.LlamaEngine` and
:class:`infra.fla_engine.FLAEngine` because Jamba is a *hybrid* model
that needs both kinds of cache simultaneously:

  * **4 attention layers** (out of 32 in v0.1, 4 of 16 in tiny-dev) use a
    paged transformer-style KV cache:
    ``[num_blocks, page_size, num_kv_heads, head_dim]`` (NHD layout),
    matching the standard kb-nano ``LlamaEngine`` cache used by
    ``flash_attn_decode``.  Each :class:`JambaAttention` module owns
    its own slice of the global cache via ``self.k_cache`` /
    ``self.v_cache`` (the engine binds these after allocation).

  * **28 Mamba layers** with per-sequence selective-scan state
    (``conv_state``: ``[num_slots, intermediate, K-1]``;
    ``ssm_state``:  ``[num_slots, intermediate, ssm_state_size]``) --
    same as :class:`infra.mamba_engine.MambaEngine` and FLAEngine.

The full vLLM v1 hybrid scheduler does paged KV + slot-allocated mamba
state with chunked prefill and CUDA graph capture.  We keep the engine
shape close to that pattern but stay single-rank and lockstep-batched
(``max_num_seqs`` rows per batch, all rows decode the same number of
steps within a batch).  Continuous batching is a follow-up.

Layout: every batch is ``[B, T]`` with left padding.  Prefill runs
batched dense attention against a left-pad-aware mask (already at
~SOTA parity); the same pass *side-writes* K/V to the paged cache via
``StoreKVCache`` so that decode can read them through
``FlashAttnDecode`` with paged ``block_table`` + ``cache_seqlens``.
This eliminates the dense-slab attention's "wasted-tail" compute (the
old design attended to all ``graph_max_total`` slots regardless of
``cur_len``) and is the change that closes the v0.1 decode-heavy gap.

The Mamba layers continue to use the flat-varlen kernel API
(``query_start_loc``, ``cache_indices``) so we reuse vLLM's SOTA Mamba
kernels directly.

Per-step state is published on the global ``Context``
(``infra/context.py``) via ``set_jamba_context`` and consumed by
:class:`L2.jamba_attention.JambaAttention` (via the standard
``slot_mapping`` / ``block_tables`` / ``context_lens`` /
``prefill_attn_mask`` fields) and :class:`L2.jamba_mamba_mixer.JambaMambaMixer`
(via the standard ``mamba_state`` / ``mamba_metadata`` fields).
This keeps Jamba's ``forward`` signature aligned with the rest of the
codebase: ``model(input_ids, positions)`` returns hidden states, with
all per-step plumbing flowing through the shared Context.

Tensor parallel: NOT supported -- single-GPU only.  Open Jamba models
fit on a B200 (Jamba-tiny-dev = 318M, Jamba-v0.1 = 52B).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from .context import get_attn_backend_config, reset_context, set_jamba_context

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


# Paged-KV page size and layout are auto-detected from the
# ``ATTN_BACKEND_CONFIG``: TRTLLM (Blackwell sm_100+) uses HND with
# block_size=16; FA3/FA2 elsewhere uses NHD with block_size=256.  We
# read the config inside ``__init__`` so engine state matches whatever
# the JambaAttention modules will dispatch to.


# ---------------------------------------------------------------------------
# Mamba metadata (unchanged from prior implementation; the Mamba mixer
# already uses flat-varlen kernels and reads from the standard
# ``mamba_metadata`` Context field).
# ---------------------------------------------------------------------------
@dataclass
class JambaMambaMetadata:
    """Per-step Mamba metadata for Jamba.

    Mirrors vLLM ``Mamba1AttentionMetadata`` (fields named to match the
    flat-varlen kernel API).  Per-layer state slabs are owned by the
    engine; layer indices on the mixer pick which slab to consume.

    ``conv_states[i]``: ``[num_slots, intermediate, K-1]`` with
                       ``stride(intermediate) == 1``.
    ``ssm_states[i]``:  ``[num_slots, intermediate, ssm_state_size]``.
    """
    conv_states: list[torch.Tensor]
    ssm_states: list[torch.Tensor]
    cache_indices: torch.Tensor          # int32 [num_seqs]
    is_decode: bool = True
    query_start_loc: torch.Tensor | None = None    # int32 [num_seqs+1] (prefill)
    has_initial_state: torch.Tensor | None = None  # bool [num_seqs] (prefill)
    pad_mask_flat: torch.Tensor | None = None      # bool [total_tokens] (prefill)


# ---------------------------------------------------------------------------
# CUDA-graph entry: holds the captured graph + the static-identity tensors
# that callers mutate in-place between replays.  All tensors here have
# stable storage; only their values are updated each step.
# ---------------------------------------------------------------------------
@dataclass
class _JambaDecodeGraph:
    graph: torch.cuda.CUDAGraph
    step_input_ids: torch.Tensor    # [B, 1]   int64 -- previous step's token
    slot_mapping: torch.Tensor      # [B]      int64 -- paged cache slot
    context_lens: torch.Tensor      # [B]      int32 -- valid len after write
    next_tokens: torch.Tensor       # [B]      int64 -- argmax output


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

        # Auto-detect attention backend & paged-cache layout.  TRTLLM-gen
        # (Blackwell, sm_100+) uses HND with block_size=16; FA3/FA2 uses
        # NHD with block_size=256.  This must match the dispatcher in
        # :class:`JambaAttention.__init__`.
        attn_cfg = get_attn_backend_config()
        self._page_size = attn_cfg.block_size
        self._kv_layout = attn_cfg.kv_layout  # "HND" or "NHD"
        self._use_trtllm = attn_cfg.use_trtllm

        # ------------------------------------------------------------------
        # CUDA-graph buckets and per-bucket static buffers.
        # ------------------------------------------------------------------
        self.graph_max_total = graph_max_total
        # blocks_per_seq covers the full ``graph_max_total`` (the longest
        # decode sequence we capture).  Round up so the last token's
        # slot never spills past the allocated blocks.
        self._blocks_per_seq = (graph_max_total + self._page_size - 1) // self._page_size

        # CUDA-graph bucket selection.  Default to single-bucket capture
        # at ``max_num_seqs``: the trailing micro-batch is padded up to
        # this size (with real-prompt clones, see ``_run_batch_graph``)
        # so the decode kernels see uniform shape.
        #
        # We had multi-bucket capture briefly (e.g. ``[24, 32]``) to
        # reduce trailing-batch padding, but capturing two graphs whose
        # paged-KV cache views point into the same global allocation
        # corrupts replay state on Blackwell + TRTLLM-gen even with a
        # shared mempool (probably a kernel-level constraint we don't
        # fully understand yet -- vLLM gets away with it via its own
        # ``CUDAGraphCapturer``).  Keeping single-bucket is the safe
        # default; a future commit can revisit multi-bucket once we
        # understand the corruption.  Override via
        # ``KB_NANO_JAMBA_BUCKETS=24,32`` env at your own risk.
        env_buckets = os.environ.get("KB_NANO_JAMBA_BUCKETS")
        if env_buckets:
            buckets = sorted({int(x) for x in env_buckets.split(",") if x.strip()})
        else:
            buckets = [self.max_num_seqs]
        if self.max_num_seqs not in buckets:
            buckets.append(self.max_num_seqs)
            buckets = sorted(set(buckets))
        self._decode_graph_buckets = buckets

        # ------------------------------------------------------------------
        # Allocate the paged KV cache.  One global ``[2, num_attn_layers,
        # num_blocks, page_size, num_kv_heads, head_dim]`` tensor; each
        # JambaAttention layer gets bound to its slice (k=cache[0,i],
        # v=cache[1,i]).  The block pool is partitioned across buckets:
        # bucket B uses blocks ``[bucket_offset[B] : bucket_offset[B] +
        # B * blocks_per_seq]``.  Within a bucket, each row i uses the
        # contiguous range ``[base + i * blocks_per_seq : base + (i+1) *
        # blocks_per_seq]``.  Block tables are static and never change.
        # ------------------------------------------------------------------
        bps = self._blocks_per_seq
        bucket_offsets: dict[int, int] = {}
        offset = 0
        for B in buckets:
            bucket_offsets[B] = offset
            offset += B * bps
        total_blocks = offset
        self._bucket_offsets = bucket_offsets
        self._num_blocks = total_blocks

        # Layout NHD: [num_blocks, page_size, num_kv_heads, head_dim] (FA3 path)
        # Layout HND: [num_blocks, num_kv_heads, page_size, head_dim] (TRTLLM path)
        # We pack K and V together in the leading dim so all attention
        # layers share one big allocation -- mirrors LlamaEngine's
        # ``self.kv_cache`` global tensor (see ``allocate_kv_cache`` in
        # ``infra/engine.py``).
        if self._kv_layout == "HND":
            self._kv_cache = torch.zeros(
                2, self._n_attn_layers, total_blocks,
                self._n_kv_heads, self._page_size, self._head_dim,
                dtype=dtype, device=self.device,
            )
        else:
            self._kv_cache = torch.zeros(
                2, self._n_attn_layers, total_blocks,
                self._page_size, self._n_kv_heads, self._head_dim,
                dtype=dtype, device=self.device,
            )
        # Bind per-layer cache views to each JambaAttention module.
        attn_modules = []
        for layer in self.model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn_modules.append(attn)
        assert len(attn_modules) == self._n_attn_layers, (
            f"Expected {self._n_attn_layers} attention modules, "
            f"found {len(attn_modules)}"
        )
        for i, attn in enumerate(attn_modules):
            attn.k_cache = self._kv_cache[0, i]
            attn.v_cache = self._kv_cache[1, i]
        self._attn_modules = attn_modules

        # ------------------------------------------------------------------
        # Precompute static block_tables for each bucket.  block_tables[B]
        # is a [B, blocks_per_seq] int32 tensor whose values are FIXED
        # for the lifetime of the engine (the block pool partition is
        # static).  Only ``slot_mapping`` and ``context_lens`` change
        # per step.
        # ------------------------------------------------------------------
        self._block_tables_per_bucket: dict[int, torch.Tensor] = {}
        for B in buckets:
            base = bucket_offsets[B]
            bt = torch.empty(B, bps, dtype=torch.int32, device=self.device)
            for i in range(B):
                row_base = base + i * bps
                bt[i].copy_(torch.arange(
                    row_base, row_base + bps, dtype=torch.int32, device=self.device,
                ))
            self._block_tables_per_bucket[B] = bt

        # Per-bucket graph state and static buffers, allocated lazily
        # below in ``_get_or_alloc_static_buffers``.
        self._decode_graphs: dict[int, "_JambaDecodeGraph"] = {}
        self._decode_graph_buffers: dict[int, dict] = {}
        # Shared CUDA-graph mempool -- vLLM-style.  Without this, each
        # ``torch.cuda.CUDAGraph()`` allocates a separate private pool,
        # and when we capture multiple bucket sizes in sequence the
        # second pool overlaps the first's reserved region; replays
        # from the smaller bucket then hit ``cudaErrorIllegalAddress``.
        # ``graph_pool_handle()`` creates a new shared pool we hand to
        # every bucket capture so they share one address space.
        self._cuda_graph_mempool_id = torch.cuda.graph_pool_handle()

        # Disable graph capture if requested (e.g. for debugging,
        # profiling the eager path, or when the CUDA driver disallows
        # graph capture in the current process).
        self._use_cuda_graphs = (
            os.environ.get("KB_NANO_JAMBA_CUDA_GRAPHS", "1") not in ("0", "false", "False")
        )
        # ``torch.compile``-fuse the decode forward.  Inductor fuses the
        # 100k-launch elementwise tail (RMSNorm chains, residual adds,
        # SwiGLU gate*up, MoE softmax) into a small number of Triton
        # kernels.  See ``_capture_decode_graph`` for the wrapping
        # function and the rationale.
        self._use_compile = (
            os.environ.get("KB_NANO_JAMBA_COMPILE", "1") not in ("0", "false", "False")
        )
        self._compiled_decode_step: callable | None = None

        # Pre-capture the decode graphs immediately so each graph's
        # private memory pool is reserved BEFORE the caching allocator
        # has a chance to fragment GPU memory with prefill / mask
        # tensors on subsequent ``_run_batch`` calls.  Skipping this
        # leads to ``cudaErrorIllegalAddress`` on graph replay when
        # the allocator hands out a block from inside the captured
        # graph's reserved region.  We capture in increasing-B order so
        # later buckets see a memory state similar to the first capture.
        if self._use_cuda_graphs:
            print(
                f"  [JambaEngine] Capturing decode graphs at "
                f"B={self._decode_graph_buckets} "
                f"(paged KV: {total_blocks} blocks x {self._page_size} = "
                f"{total_blocks * self._page_size} token slots, "
                f"{self._n_attn_layers} attn layers)"
            )
            for bucket in self._decode_graph_buckets:
                self._capture_decode_graph(bucket)

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
    # Static (per-bucket) decode buffers.  Allocated lazily on first use
    # and reused across ``_run_batch`` calls so the captured CUDA graph's
    # tensor pointers stay valid.  The contents are reset at the start
    # of each batch.
    # ------------------------------------------------------------------
    def _get_or_alloc_static_buffers(self, batch_size: int) -> dict:
        if batch_size in self._decode_graph_buffers:
            return self._decode_graph_buffers[batch_size]

        device = self.device

        # Mamba state slabs (per mamba layer).  Same as the prior
        # implementation: per-row slot-based state.
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
        step_positions = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device,
        )
        # Paged-KV decode metadata (static identity; values updated
        # in-place each step before ``graph.replay()``).
        slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=device)
        context_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        block_tables = self._block_tables_per_bucket[batch_size]
        next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

        bufs = {
            "B": batch_size,
            "conv_states": conv_states,
            "ssm_states": ssm_states,
            "cache_indices": cache_indices,
            "step_input_ids": step_input_ids,
            "step_positions": step_positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "next_tokens": next_tokens,
        }
        self._decode_graph_buffers[batch_size] = bufs
        return bufs

    def _capture_decode_graph(self, batch_size: int) -> _JambaDecodeGraph:
        """Capture (or replay-cached) the decode-step CUDA graph at bucket B.

        The graph reads from the static buffers in
        ``_decode_graph_buffers[B]`` and writes the sampled token to
        ``next_tokens``.  The host loop in ``_run_batch_graph`` must,
        for each step:

          1. Update ``step_input_ids[:,0]`` (in-place) to the previous
             step's output token (from ``next_tokens``).
          2. Update ``slot_mapping[:]`` to the paged-cache slot for the
             new token's position (computed from ``block_tables`` +
             ``cur_len``).
          3. Update ``context_lens[:]`` to ``cur_len + 1`` (so paged
             attention attends to the just-written K/V).
          4. ``graph.replay()``.
          5. (Async) read ``next_tokens`` into a host history buffer.

        We build the per-step ``set_jamba_context`` install ONCE up front
        (the tensor identities are stable across replays; only their
        contents are mutated) and call it *inside* the captured region,
        the same way :class:`infra.engine.ModelRunner.capture_mamba_cudagraph`
        installs its Mamba context inside its capture region.
        """
        if batch_size in self._decode_graphs:
            return self._decode_graphs[batch_size]

        bufs = self._get_or_alloc_static_buffers(batch_size)
        bps = self._blocks_per_seq
        max_context_len = bps * self._page_size

        mamba_meta = JambaMambaMetadata(
            conv_states=bufs["conv_states"],
            ssm_states=bufs["ssm_states"],
            cache_indices=bufs["cache_indices"],
            is_decode=True,
        )

        # ``torch.compile`` the decode forward.  We compile the inner
        # ``JambaModel`` + ``lm_head`` + ``argmax`` as a single function
        # so Inductor can see the full elementwise tail end-to-end
        # (RMSNorm + residual + SwiGLU pieces).  Graph breaks happen
        # automatically at the vLLM Mamba kernel calls (opaque to Dynamo)
        # and at ``get_context()`` lookups; the *between-break* regions
        # are exactly where the 100k-launch elementwise overhead lives.
        if self._use_compile and self._compiled_decode_step is None:
            inner = self.model.model
            lm_head = self.model.lm_head

            def _forward_for_compile(
                input_ids: torch.Tensor,
                positions: torch.Tensor,
            ) -> torch.Tensor:
                hidden = inner(input_ids, positions)
                logits = lm_head(hidden[:, -1, :])
                # ``argmax`` on bf16 is fine; we don't need .float() here
                # because argmax is dtype-invariant.
                return logits.argmax(dim=-1)

            # Bump Dynamo cache limits.  Jamba has 32 layers; each layer
            # has a different ``self.layer_idx`` int attribute, so Dynamo
            # treats every layer as a fresh compile target.  The default
            # ``recompile_limit=8`` causes later layers to silently fall
            # back to eager.  Lift to a comfortable margin over
            # ``num_hidden_layers``.
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit,
                self.config.num_hidden_layers * 2 + 32,
            )
            torch._dynamo.config.accumulated_cache_size_limit = max(
                torch._dynamo.config.accumulated_cache_size_limit,
                self.config.num_hidden_layers * 4 + 64,
            )
            if hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
                torch._dynamo.config.allow_unspec_int_on_nn_module = True

            self._compiled_decode_step = torch.compile(
                _forward_for_compile,
                mode="default",
                dynamic=False,
                fullgraph=False,
            )

        # Closure: one decode step, fully static-shape.
        def _decode_step():
            set_jamba_context(
                is_prefill=False,
                slot_mapping=bufs["slot_mapping"],
                context_lens=bufs["context_lens"],
                block_tables=bufs["block_tables"],
                max_context_len=max_context_len,
                mamba_state=None,
                mamba_metadata=mamba_meta,
            )
            try:
                if self._compiled_decode_step is not None:
                    tok = self._compiled_decode_step(
                        bufs["step_input_ids"],
                        bufs["step_positions"],
                    )
                else:
                    hidden = self.model(
                        bufs["step_input_ids"],
                        bufs["step_positions"],
                    )
                    logits = self.model.compute_logits(hidden[:, -1, :])
                    tok = logits.argmax(dim=-1)
                # Write into the persistent buffer.  Must be in-place so
                # the graph's output tensor identity is fixed.
                bufs["next_tokens"].copy_(tok)
            finally:
                reset_context()

        # Warmup outside the graph-capture stream to populate workspace
        # tensors / autotune caches.  Using a fresh stream that's joined
        # back to the current stream ensures all allocator state is
        # settled before capture begins.
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _decode_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # Drop any cached allocator blocks that might overlap with the
        # graph's private pool (avoids ``illegal memory access`` on
        # replay if the allocator hands out a block from inside a
        # previously-captured graph's reserved pool).
        torch.cuda.empty_cache()

        graph = torch.cuda.CUDAGraph()
        # Share the mempool across all bucket graphs so their address
        # spaces don't collide on replay.  See the ``mempool_id``
        # comment in ``__init__``.
        with torch.cuda.graph(graph, pool=self._cuda_graph_mempool_id):
            _decode_step()

        entry = _JambaDecodeGraph(
            graph=graph,
            step_input_ids=bufs["step_input_ids"],
            slot_mapping=bufs["slot_mapping"],
            context_lens=bufs["context_lens"],
            next_tokens=bufs["next_tokens"],
        )
        self._decode_graphs[batch_size] = entry
        return entry

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

    @staticmethod
    def _make_positions(B: int, T: int, device, attention_mask=None) -> torch.Tensor:
        """Build a ``[B, T]`` positions tensor.  Jamba's mixers ignore
        positions (no RoPE; Mamba carries position via recurrence) but
        we produce them anyway so the forward signature aligns with
        Llama / Mamba / Mixtral.
        """
        if attention_mask is not None:
            pos = attention_mask.long().cumsum(dim=-1) - 1
            return pos.clamp_(min=0)
        return torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)

    # ------------------------------------------------------------------
    # Build flat-varlen prefill metadata for a left-padded batch.
    #
    # Returns a dict with:
    #   slot_mapping: [N] int64 (one entry per real token; values are
    #                 paged-cache slot IDs)
    #   cu_seqlens_q: [B+1] int32 (cumulative real-token counts, kernel
    #                 batching descriptor)
    #   cu_seqlens_k: [B+1] int32 (same as cu_seqlens_q for prefill --
    #                 K and Q are the same flat layout)
    #   flat_to_grid: [N] int64 (the i-th real token came from row r,
    #                 col c in the [B, T_max] grid; index = r*T_max + c)
    #   max_seqlen:   int (= max prompt length in the batch)
    #
    # Computed on host with vectorized numpy then transferred in one
    # H2D copy per tensor -- amortized cost is small relative to prefill.
    # ------------------------------------------------------------------
    def _build_prefill_metadata(
        self,
        B_pad: int,
        max_prompt: int,
        prompt_lens: list[int],
        block_tables: torch.Tensor,
    ) -> dict:
        device = self.device
        plens = np.array(prompt_lens, dtype=np.int64)              # [B]
        offsets = max_prompt - plens                                # [B] left-pad
        total = int(plens.sum())

        # cu_seqlens (Q == K for prefill).
        cu_q = np.zeros(B_pad + 1, dtype=np.int32)
        np.cumsum(plens, out=cu_q[1:])

        # flat_to_grid[k] = the [B*T_max] flat index of real-token k.
        # For row i with prompt_lens[i] = L, real tokens occupy
        # [offsets[i], offsets[i] + L) in the [T_max] axis, and
        # [i*T_max + offsets[i], i*T_max + offsets[i] + L) in the
        # flat [B*T_max] view.
        flat_to_grid = np.empty(total, dtype=np.int64)
        # slot_mapping[k] = paged cache slot for the k-th real token.
        slot_mapping = np.empty(total, dtype=np.int64)
        bt_host = block_tables.cpu().numpy()                        # [B, bps]
        idx = 0
        for i, plen in enumerate(prompt_lens):
            grid_start = i * max_prompt + int(offsets[i])
            for j in range(plen):
                flat_to_grid[idx] = grid_start + j
                block_idx = j // self._page_size
                slot_in_block = j % self._page_size
                slot_mapping[idx] = (
                    int(bt_host[i, block_idx]) * self._page_size + slot_in_block
                )
                idx += 1

        return {
            "slot_mapping": torch.from_numpy(slot_mapping).to(device),
            "cu_seqlens": torch.from_numpy(cu_q).to(device),
            "flat_to_grid": torch.from_numpy(flat_to_grid).to(device),
            "max_seqlen": int(plens.max()),
            "total_real_tokens": total,
        }

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

        # CUDA-graph eligibility: greedy + fits in static graph_max_total.
        all_greedy = all(p.temperature == 0.0 for p in sampling_params)
        fits_static = max_total <= self.graph_max_total
        use_graph = self._use_cuda_graphs and all_greedy and fits_static

        # Build left-padded prompt tensor + attention mask (1=real, 0=pad).
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
                prompt_lens=prompt_lens, pad_id=pad_id,
            )

        # ==============================================================
        # Eager fallback (non-greedy, or CUDA graphs disabled).  Uses the
        # B == max_num_seqs bucket's static buffers (paged cache + Mamba
        # state) and runs the decode loop without graph replay.
        # ==============================================================
        return self._run_batch_eager(
            prompt_token_ids, sampling_params,
            B=B, max_prompt=max_prompt, max_out=max_out, max_total=max_total,
            input_ids=input_ids, attention_mask=attention_mask,
            prompt_lens=prompt_lens, pad_id=pad_id,
        )

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
        prompt_lens: list[int],
        pad_id: int,
    ) -> list[GenerationOutput]:
        device = self.device
        eos = self.tokenizer.eos_token_id
        # Pick the smallest bucket >= B (the actual batch size).
        B_pad = self._pick_bucket(B)

        # Pad prompts up to B_pad by *cloning* the first real prompt
        # into the trailing rows.  We tried a "dummy" pad with
        # ``prompt_lens=1`` first, but the TRTLLM-gen paged decode
        # kernel crashes during graph replay when the batch contains
        # rows with widely-divergent ``cache_seqlens`` (real rows at
        # ~1k tokens vs dummy rows at ~1 token).  Cloning the first
        # real prompt makes ``cache_seqlens`` uniform across all rows
        # in the bucket, which:
        #   - sidesteps the TRTLLM kernel grid issue,
        #   - matches what the captured graph saw at warmup time,
        #   - has the same throughput cost as a dummy pad (both
        #     occupy a graph slot for the duration of decode), and
        #   - the padded rows' outputs are discarded after generate.
        if B < B_pad:
            extra = B_pad - B
            row0 = input_ids[0:1]                    # [1, max_prompt]
            attn0 = attention_mask[0:1]              # [1, max_prompt]
            input_ids = torch.cat(
                [input_ids, row0.expand(extra, -1).contiguous()], dim=0,
            )
            attention_mask = torch.cat(
                [attention_mask, attn0.expand(extra, -1).contiguous()], dim=0,
            )
            prompt_lens = prompt_lens + [prompt_lens[0]] * extra

        # Static buffers + graph entry for this bucket.
        graph_entry = self._capture_decode_graph(B_pad)
        bufs = self._get_or_alloc_static_buffers(B_pad)
        conv_states = bufs["conv_states"]
        ssm_states = bufs["ssm_states"]
        cache_indices = bufs["cache_indices"]
        step_input_ids = bufs["step_input_ids"]
        slot_mapping = bufs["slot_mapping"]
        context_lens = bufs["context_lens"]
        block_tables = bufs["block_tables"]
        next_tokens_buf = bufs["next_tokens"]

        # Reset Mamba state to zero for a clean slate; KV cache slots
        # are overwritten by the prefill side-write so no zeroing
        # needed for those.
        for cs in conv_states:
            cs.zero_()
        for ss in ssm_states:
            ss.zero_()

        # ---- Prefill ------------------------------------------------------
        # Paged-context attention (TRTLLM-gen on Blackwell, FA3 elsewhere)
        # via flat-varlen Q/K/V.  ``_build_prefill_metadata`` produces the
        # cu_seqlens / flat_to_grid / slot_mapping the L2 attention uses
        # to remap [B, T_max, h] left-padded → flat-varlen → [B, T_max, h].
        pmeta = self._build_prefill_metadata(
            B_pad, max_prompt, prompt_lens, block_tables,
        )

        mamba_qsl_p = torch.tensor(
            [i * max_prompt for i in range(B_pad + 1)],
            dtype=torch.int32, device=device,
        )
        mamba_has_init = torch.zeros(B_pad, dtype=torch.bool, device=device)
        mamba_pad_flat = attention_mask.bool().reshape(-1)

        positions = self._make_positions(B_pad, max_prompt, device, attention_mask)

        mamba_meta_p = JambaMambaMetadata(
            conv_states=conv_states,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            is_decode=False,
            query_start_loc=mamba_qsl_p,
            has_initial_state=mamba_has_init,
            pad_mask_flat=mamba_pad_flat,
        )
        set_jamba_context(
            is_prefill=True,
            slot_mapping=pmeta["slot_mapping"],
            block_tables=block_tables,
            cu_seqlens_q=pmeta["cu_seqlens"],
            cu_seqlens_k=pmeta["cu_seqlens"],
            max_seqlen_q=pmeta["max_seqlen"],
            max_seqlen_k=pmeta["max_seqlen"],
            flat_to_grid=pmeta["flat_to_grid"],
            mamba_metadata=mamba_meta_p,
        )
        try:
            hidden = self.model(input_ids, positions)
            prefill_logits = self.model.compute_logits(hidden[:, -1, :])
        finally:
            reset_context()
        first_tok = prefill_logits.argmax(dim=-1)  # [B_pad]

        # ---- Decode loop driven by graph replay --------------------------
        per_row_max = [p.max_tokens for p in sampling_params]
        global_max = max_out

        tok_history = torch.empty(
            global_max, B_pad, dtype=torch.long, pin_memory=True,
        )
        tok_history[0].copy_(first_tok, non_blocking=True)
        step_input_ids[:, 0].copy_(first_tok)

        # Per-row prompt length (used for paged-KV slot computation).
        # Each row's "context_len after this step" depends on its own
        # prompt length, since rows have different lengths under
        # left-padding.  After the first decode token is sampled, row
        # i has ``prompt_lens[i]`` real K/V slots already written; the
        # next K/V (for the new token) goes to slot ``prompt_lens[i]``,
        # and the post-write ``context_lens[i] = prompt_lens[i] + 1``.
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.int32, device=device)

        # Pre-build per-step slot_mapping table on host: slot_table[s,i]
        # = block_tables[i, (prompt_lens[i] + s) // page_size] *
        #   page_size + ((prompt_lens[i] + s) % page_size)
        # for s in [0, global_max - 1].  This is a [global_max, B_pad]
        # int64 lookup we update slot_mapping/context_lens from each step.
        bt_host = block_tables.cpu().numpy()
        slot_table = np.empty((global_max, B_pad), dtype=np.int64)
        for s in range(global_max):
            for i in range(B_pad):
                pos = prompt_lens[i] + s
                block_idx = pos // self._page_size
                slot_in_block = pos % self._page_size
                slot_table[s, i] = (
                    int(bt_host[i, block_idx]) * self._page_size + slot_in_block
                )
        slot_table_t = torch.from_numpy(slot_table).to(device)
        # context_lens after writing decode step ``s`` (0-indexed) is
        # ``prompt_lens[i] + s + 1`` for each row.
        ctxlen_table = (
            prompt_lens_t.unsqueeze(0) + torch.arange(
                global_max, device=device, dtype=torch.int32,
            ).unsqueeze(1) + 1
        ).contiguous()

        # Step 0: write the first decode token's K/V at slot
        # ``prompt_lens[i]`` per row -- but the model forward needs
        # context_lens to reflect post-write length.  So we set
        # context_lens for step 0 = prompt_lens + 1 BEFORE calling
        # graph.replay().
        for step in range(1, global_max):
            slot_mapping.copy_(slot_table_t[step - 1])
            context_lens.copy_(ctxlen_table[step - 1])
            graph_entry.graph.replay()
            tok_history[step].copy_(next_tokens_buf, non_blocking=True)
            step_input_ids[:, 0].copy_(next_tokens_buf)

        torch.cuda.synchronize()

        # ---- Build per-row generated lists (drop padding rows) -----------
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

    # ------------------------------------------------------------------
    # Eager fallback (non-greedy, or graphs disabled).  Mirrors the graph
    # path but without the captured graph; uses a fresh per-batch context
    # install on each step.  Same paged-KV plumbing.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _run_batch_eager(
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
        prompt_lens: list[int],
        pad_id: int,
    ) -> list[GenerationOutput]:
        device = self.device
        eos = self.tokenizer.eos_token_id
        B_pad = self._pick_bucket(B)
        if B < B_pad:
            # Pad with clones of the first real prompt (same approach as
            # the graph path) so all rows have uniform cache_seqlens for
            # the paged decode kernel.
            extra = B_pad - B
            row0 = input_ids[0:1]
            attn0 = attention_mask[0:1]
            input_ids = torch.cat(
                [input_ids, row0.expand(extra, -1).contiguous()], dim=0,
            )
            attention_mask = torch.cat(
                [attention_mask, attn0.expand(extra, -1).contiguous()], dim=0,
            )
            prompt_lens = prompt_lens + [prompt_lens[0]] * extra

        # Reuse the bucket's static buffers for Mamba state + paged-KV
        # block_tables.  Slot mapping / context_lens are rebuilt per step.
        bufs = self._get_or_alloc_static_buffers(B_pad)
        conv_states = bufs["conv_states"]
        ssm_states = bufs["ssm_states"]
        cache_indices = bufs["cache_indices"]
        block_tables = bufs["block_tables"]
        for cs in conv_states:
            cs.zero_()
        for ss in ssm_states:
            ss.zero_()

        pmeta = self._build_prefill_metadata(
            B_pad, max_prompt, prompt_lens, block_tables,
        )
        mamba_qsl_p = torch.tensor(
            [i * max_prompt for i in range(B_pad + 1)],
            dtype=torch.int32, device=device,
        )
        mamba_has_init = torch.zeros(B_pad, dtype=torch.bool, device=device)
        mamba_pad_flat = attention_mask.bool().reshape(-1)
        positions = self._make_positions(B_pad, max_prompt, device, attention_mask)

        mamba_meta_p = JambaMambaMetadata(
            conv_states=conv_states,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            is_decode=False,
            query_start_loc=mamba_qsl_p,
            has_initial_state=mamba_has_init,
            pad_mask_flat=mamba_pad_flat,
        )
        set_jamba_context(
            is_prefill=True,
            slot_mapping=pmeta["slot_mapping"],
            block_tables=block_tables,
            cu_seqlens_q=pmeta["cu_seqlens"],
            cu_seqlens_k=pmeta["cu_seqlens"],
            max_seqlen_q=pmeta["max_seqlen"],
            max_seqlen_k=pmeta["max_seqlen"],
            flat_to_grid=pmeta["flat_to_grid"],
            mamba_metadata=mamba_meta_p,
        )
        try:
            hidden = self.model(input_ids, positions)
            logits = self.model.compute_logits(hidden[:, -1, :])
        finally:
            reset_context()
        next_tokens = self._sample_step(logits, sampling_params + [SamplingParams()] * (B_pad - B))
        # Truncate/pad to B_pad to match buffer shapes.

        generated: list[list[int]] = [[] for _ in range(B_pad)]
        finished = [False] * B_pad
        for i, t in enumerate(next_tokens):
            generated[i].append(int(t))
            sp = sampling_params[i] if i < B else SamplingParams()
            if not sp.ignore_eos and eos is not None and t == eos:
                finished[i] = True
            if len(generated[i]) >= sp.max_tokens:
                finished[i] = True

        bps = self._blocks_per_seq
        max_context_len = bps * self._page_size

        cur_step = 1
        slot_mapping_t = torch.zeros(B_pad, dtype=torch.long, device=device)
        context_lens_t = torch.zeros(B_pad, dtype=torch.int32, device=device)
        bt_host = block_tables.cpu().numpy()
        prompt_lens_arr = np.array(prompt_lens, dtype=np.int64)

        while cur_step < max_out and not all(finished[:B]):
            step_ids = torch.tensor(
                [[generated[i][-1]] for i in range(B_pad)],
                dtype=torch.long, device=device,
            )
            step_positions = torch.tensor(
                [[prompt_lens[i] + cur_step - 1] for i in range(B_pad)],
                dtype=torch.long, device=device,
            )
            # Per-step slot_mapping + context_lens.
            slots_host = np.empty(B_pad, dtype=np.int64)
            ctx_host = np.empty(B_pad, dtype=np.int32)
            for i in range(B_pad):
                pos = prompt_lens[i] + cur_step - 1 + 1  # post-write context len
                slot_pos = pos - 1
                block_idx = slot_pos // self._page_size
                slot_in_block = slot_pos % self._page_size
                slots_host[i] = int(bt_host[i, block_idx]) * self._page_size + slot_in_block
                ctx_host[i] = pos
            slot_mapping_t.copy_(torch.from_numpy(slots_host))
            context_lens_t.copy_(torch.from_numpy(ctx_host))

            mamba_meta_d = JambaMambaMetadata(
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                is_decode=True,
            )
            set_jamba_context(
                is_prefill=False,
                slot_mapping=slot_mapping_t,
                context_lens=context_lens_t,
                block_tables=block_tables,
                max_context_len=max_context_len,
                mamba_metadata=mamba_meta_d,
            )
            try:
                hidden = self.model(step_ids, step_positions)
                logits = self.model.compute_logits(hidden[:, -1, :])
            finally:
                reset_context()
            next_tokens = self._sample_step(
                logits, sampling_params + [SamplingParams()] * (B_pad - B),
            )
            for i, t in enumerate(next_tokens):
                if finished[i]:
                    continue
                generated[i].append(int(t))
                sp = sampling_params[i] if i < B else SamplingParams()
                if not sp.ignore_eos and eos is not None and t == eos:
                    finished[i] = True
                if len(generated[i]) >= sp.max_tokens:
                    finished[i] = True
            cur_step += 1

        # Trim padded rows.
        generated_real = generated[:B]
        return self._materialise(generated_real)

    def _pick_bucket(self, B: int) -> int:
        """Return the smallest captured decode-graph bucket >= ``B``.

        Falls back to ``max_num_seqs`` if every bucket is smaller (which
        shouldn't happen because ``max_num_seqs`` is always in the bucket
        list).  Mirrors vLLM's ``_graph_bs_for_n`` lookup.
        """
        for b in self._decode_graph_buckets:
            if b >= B:
                return b
        return self.max_num_seqs

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
