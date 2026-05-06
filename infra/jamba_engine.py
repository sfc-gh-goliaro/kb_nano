"""Jamba inference engine.

Single-rank serving engine for AI21Labs Jamba (triple-hybrid Transformer
+ Mamba-1 + sparse MoE).  Mirrors :class:`infra.engine.LlamaEngine`'s
serving pattern -- paged KV cache for the attention layers + slot-based
state for the Mamba layers -- so the L2 modules
(:class:`L2.jamba_attention.JambaAttention`,
:class:`L2.jamba_mamba_mixer.JambaMambaMixer`) match vLLM's interface
(fused ``QKVParallelLinear`` + the project's ``Attention`` class +
``RowParallelLinear``; flat varlen ``[N, hidden]`` throughout) and can
be ported into vLLM directly.

Pipeline layout (matches vLLM):

  * ``input_ids``: flat int64 ``[N]`` where ``N = sum(prompt_lens)``
    for prefill or ``N = num_active_seqs`` for decode.
  * ``positions``: flat int64 ``[N]`` (unused by Jamba mixers but threaded
    through for signature uniformity with Llama / Mamba).
  * Per-step paged-KV metadata flows through ``set_jamba_context`` ->
    standard ``Context`` fields:
      - ``slot_mapping``: ``[N]`` int64, paged-cache slot per token.
      - ``block_tables``: ``[B, max_blocks_per_seq]`` int32.
      - ``context_lens``: ``[B]`` int32 (decode only).
      - ``cu_seqlens_q`` / ``cu_seqlens_k``: ``[B+1]`` int32 (prefill).
      - ``max_seqlen_q`` / ``max_seqlen_k``: int.
  * Mamba state (per-sequence) flows via ``mamba_state`` /
    ``mamba_metadata`` Context fields.

Cache allocation (mirrors ``LlamaEngine.allocate_kv_cache``):

  * One global ``[2, num_attn_layers, num_blocks, page_size,
    num_kv_heads, head_dim]`` (NHD) or ``[2, num_attn_layers, num_blocks,
    num_kv_heads, page_size, head_dim]`` (HND) tensor; each
    :class:`L2.attention_impl.Attention` gets bound to its slice via
    ``module.k_cache = kv_cache[0, i]`` / ``module.v_cache = kv_cache[1, i]``.
  * Per-sequence Mamba state slabs sized for ``max_num_seqs`` slots;
    sequence ``s`` uses slot ``s.state_slot`` in every Mamba layer.

Scheduler: lockstep micro-batching at ``max_num_seqs``.  Each batch
runs full prefill once, then loops single-step decode until every
sequence hits its ``max_tokens`` (or eos).  Continuous batching (admit
new sequences mid-decode) is not implemented yet; the lockstep pattern
is what the current bench tolerates and a future commit can extend
this engine with a Sequence-list scheduler à la LlamaEngine if the
profile justifies it.

Tensor parallel: NOT supported -- single-GPU only.  Open Jamba models
fit on a B200 (Jamba-tiny-dev = 318M, Jamba-v0.1 = 52B).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from .context import (
    auto_register_no_compile_layers,
    enable_custom_ops,
    get_attn_backend_config,
    register_no_compile_layers,
    reset_context,
    set_jamba_context,
)

# Re-export the same SamplingParams / GenerationOutput dataclasses the
# rest of the codebase uses.  We import lazily because ``infra.engine``
# transitively imports the entire model zoo -- JambaEngine has no need
# for any of it.  Fallback definitions match the originals
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
# Mamba metadata (per-batch, per-step).  Mirrors vLLM
# ``Mamba1AttentionMetadata`` / kb-nano's existing pattern; the L2 mamba
# mixer reads ``conv_states`` / ``ssm_states`` / ``cache_indices`` plus
# ``query_start_loc`` (prefill) or just ``cache_indices`` (decode) and
# runs the flat-varlen vendor kernels.
# ---------------------------------------------------------------------------
@dataclass
class JambaMambaMetadata:
    conv_states: list[torch.Tensor]
    ssm_states: list[torch.Tensor]
    cache_indices: torch.Tensor          # int32 [num_seqs]
    is_decode: bool = True
    query_start_loc: torch.Tensor | None = None    # int32 [num_seqs+1] (prefill)
    has_initial_state: torch.Tensor | None = None  # bool [num_seqs] (prefill)
    pad_mask_flat: torch.Tensor | None = None      # legacy; unused with flat layout


# ---------------------------------------------------------------------------
# Decode-step CUDA graph entry.  Static-identity tensors that callers
# mutate in-place between replays.
# ---------------------------------------------------------------------------
@dataclass
class _JambaDecodeGraph:
    graph: torch.cuda.CUDAGraph
    step_input_ids: torch.Tensor    # [B]      int64 -- previous step's token
    step_positions: torch.Tensor    # [B]      int64 -- absolute pos for new token
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
        max_num_seqs: int = 32,
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

        # ------------------------------------------------------------------
        # Build the model on CPU then move to GPU + cast to dtype.
        # ------------------------------------------------------------------
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

        # Move to GPU.  A and conv_state must remain fp32 / bf16 as
        # configured -- cast everything to ``dtype`` for now, then
        # restore A to fp32 (the Mamba SSM kernels require A fp32).
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        for layer in self.model.model.layers:
            mamba = getattr(layer, "mamba", None)
            if mamba is not None:
                mamba.A.data = mamba.A.data.float()

        torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # Cache shapes
        # ------------------------------------------------------------------
        cfg = self.config
        self._mamba_intermediate = cfg.mamba_expand * cfg.hidden_size
        self._mamba_d_state = cfg.mamba_d_state
        self._mamba_conv_kernel = cfg.mamba_d_conv
        self._n_mamba_layers = len(self.model.model.mamba_layer_indices)
        self._n_attn_layers = len(self.model.model.attention_layer_indices)
        self._head_dim = cfg.hidden_size // cfg.num_attention_heads
        self._n_kv_heads = cfg.num_key_value_heads

        # Auto-detect attention backend & paged-cache layout (TRTLLM-gen
        # HND on Blackwell, FA3/FA2 NHD elsewhere).  Must match the
        # dispatcher in :class:`L2.attention_impl.Attention`.
        attn_cfg = get_attn_backend_config()
        self._page_size = attn_cfg.block_size
        self._kv_layout = attn_cfg.kv_layout  # "HND" or "NHD"
        self._use_trtllm = attn_cfg.use_trtllm

        self.graph_max_total = graph_max_total
        # blocks_per_seq covers ``graph_max_total`` (the longest seq we
        # support).  Round up so the last token's slot doesn't spill.
        self._blocks_per_seq = (graph_max_total + self._page_size - 1) // self._page_size

        # ------------------------------------------------------------------
        # Paged KV cache allocation -- mirrors LlamaEngine.allocate_kv_cache.
        # One global tensor; each :class:`Attention` gets bound to a slice
        # (``module.k_cache = kv_cache[0, i]``,
        # ``module.v_cache = kv_cache[1, i]``).  Engine partitions blocks
        # per row in the bucket so each row's block_table is fixed.
        # ------------------------------------------------------------------
        bps = self._blocks_per_seq
        total_blocks = self.max_num_seqs * bps
        self._num_blocks = total_blocks

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

        # Bind per-layer cache slices to each Attention module.  The L4
        # model wraps the inner ``Attention`` as ``layer.self_attn.attn``;
        # we walk the model's attention decoder layers in physical order
        # and assign cache index ``i`` based on the order we see them
        # (matching the ``attention_layer_indices`` order used by the
        # weight loader and the L4 layer schedule).
        attn_modules = []
        no_compile_layers: dict[str, "torch.nn.Module"] = {}
        for name, mod in self.model.named_modules():
            # Inner ``Attention`` instances live at e.g.
            # ``model.layers.4.self_attn.attn`` -- detect by the
            # presence of ``k_cache`` / ``v_cache`` attributes set in
            # ``Attention.__init__``.
            if (hasattr(mod, "k_cache") and hasattr(mod, "v_cache")
                    and type(mod).__name__ == "Attention"):
                attn_modules.append(mod)
                no_compile_layers[name] = mod
        assert len(attn_modules) == self._n_attn_layers, (
            f"Expected {self._n_attn_layers} Attention instances, found "
            f"{len(attn_modules)}; check L2 JambaAttention wiring."
        )
        for i, attn in enumerate(attn_modules):
            attn.k_cache = self._kv_cache[0, i]
            attn.v_cache = self._kv_cache[1, i]
            # The Attention class stores its own layer name for custom-op
            # dispatch; populate it now (matches LlamaEngine pattern).
            attn._layer_name = next(
                n for n, m in no_compile_layers.items() if m is attn
            )
        # Register the Attention modules so torch.compile custom ops
        # can resolve them at runtime.  Mirrors LlamaEngine's
        # ``auto_register_no_compile_layers(self.model)`` call.
        register_no_compile_layers(no_compile_layers)
        auto_register_no_compile_layers(self.model)
        self._attn_modules = attn_modules

        # Static block_tables: row i in the lockstep bucket uses blocks
        # ``[i*bps, (i+1)*bps)``.  These never change.
        bt = torch.empty(
            self.max_num_seqs, bps, dtype=torch.int32, device=self.device,
        )
        for i in range(self.max_num_seqs):
            row_base = i * bps
            bt[i].copy_(torch.arange(
                row_base, row_base + bps, dtype=torch.int32, device=self.device,
            ))
        self._block_tables = bt

        # ------------------------------------------------------------------
        # Mamba state slabs -- one slot per row in the lockstep bucket.
        # Same convention as MambaEngine / FLAEngine: ``cache_indices``
        # is a fixed ``arange(max_num_seqs)`` int32 tensor.
        # ------------------------------------------------------------------
        K_minus_1 = max(self._mamba_conv_kernel - 1, 1)
        self._conv_states: list[torch.Tensor] = []
        self._ssm_states: list[torch.Tensor] = []
        for _ in range(self._n_mamba_layers):
            raw_conv = torch.zeros(
                self.max_num_seqs, K_minus_1, self._mamba_intermediate,
                dtype=dtype, device=self.device,
            )
            self._conv_states.append(raw_conv.transpose(-1, -2))
            self._ssm_states.append(torch.zeros(
                self.max_num_seqs, self._mamba_intermediate, self._mamba_d_state,
                dtype=dtype, device=self.device,
            ))
        self._cache_indices = torch.arange(
            self.max_num_seqs, dtype=torch.int32, device=self.device,
        )

        # ------------------------------------------------------------------
        # Decode static buffers.  All shapes are fixed at B=max_num_seqs
        # so the captured CUDA graph reads from stable identities.
        # ------------------------------------------------------------------
        B = self.max_num_seqs
        self._step_input_ids = torch.zeros(B, dtype=torch.long, device=self.device)
        self._step_positions = torch.zeros(B, dtype=torch.long, device=self.device)
        self._step_slot_mapping = torch.zeros(B, dtype=torch.long, device=self.device)
        self._step_context_lens = torch.zeros(B, dtype=torch.int32, device=self.device)
        self._step_next_tokens = torch.zeros(B, dtype=torch.long, device=self.device)

        # ------------------------------------------------------------------
        # CUDA graph capture / torch.compile.  Decode forward is wrapped
        # with ``torch.compile(mode="default")`` before capture so
        # Inductor fuses the elementwise tail (RMSNorm + residual +
        # SwiGLU + MoE softmax pieces) into Triton kernels.
        # ------------------------------------------------------------------
        self._use_cuda_graphs = (
            os.environ.get("KB_NANO_JAMBA_CUDA_GRAPHS", "1") not in ("0", "false", "False")
        )
        # ``torch.compile`` is OFF by default.  Inductor's fused
        # elementwise paths (RMSNorm + residual + SwiGLU pieces) produce
        # bf16 intermediates that drift from vLLM's hand-written CUDA
        # kernels by ~1e-3 per layer; over 32 layers and 128 decode
        # steps this flips greedy top-1 within ~5 tokens for most
        # prompts, dropping match-tokens vs vLLM from ~85/128 to ~25/128.
        # The eager path uses the SAME vLLM ``_C.fused_add_rms_norm`` /
        # ``_C.silu_and_mul`` kernels vLLM itself uses, so numerics are
        # bit-identical with the reference.
        #
        # Perf cost of disabling compile: small at v0.1 scale (0.89 ->
        # 0.90 avg, since the GEMMs / Mamba / MoE kernels dominate at
        # 52B and the elementwise tail is a small fraction); ~25-30%
        # at tiny-dev scale (1.85 -> 1.48 avg, where elementwise
        # dominates the 16-layer 512-hidden model).  Match-tokens
        # correctness is the higher-priority bar so eager wins by default.
        # Set ``KB_NANO_JAMBA_COMPILE=1`` to enable compile if you
        # specifically need the elementwise fusion and can tolerate
        # the bf16 drift vs vLLM.
        self._use_compile = (
            os.environ.get("KB_NANO_JAMBA_COMPILE", "0") not in ("0", "false", "False")
        )
        self._compiled_decode_step: callable | None = None
        self._decode_graph: _JambaDecodeGraph | None = None
        if self._use_cuda_graphs:
            print(
                f"  [JambaEngine] Capturing decode graph at B={self.max_num_seqs} "
                f"(paged KV: {total_blocks} blocks x {self._page_size} = "
                f"{total_blocks * self._page_size} token slots, "
                f"{self._n_attn_layers} attn layers, {self._kv_layout} layout)"
            )
            self._decode_graph = self._capture_decode_graph()

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
    # Decode CUDA-graph capture.  Captures one decode step at
    # B=max_num_seqs; host loop mutates step_input_ids/positions/
    # slot_mapping/context_lens between replays.
    # ------------------------------------------------------------------
    def _capture_decode_graph(self) -> _JambaDecodeGraph:
        B = self.max_num_seqs
        max_context_len = self._blocks_per_seq * self._page_size

        mamba_meta = JambaMambaMetadata(
            conv_states=self._conv_states,
            ssm_states=self._ssm_states,
            cache_indices=self._cache_indices,
            is_decode=True,
        )

        # ``torch.compile`` the decode forward.  We compile the inner
        # ``JambaModel`` + ``lm_head`` + ``argmax`` so Inductor can see
        # the full elementwise tail end-to-end (RMSNorm + residual +
        # SwiGLU pieces).  Graph breaks happen automatically at vLLM
        # mamba kernel calls and at ``get_context()`` lookups.
        if self._use_compile and self._compiled_decode_step is None:
            inner = self.model.model
            lm_head = self.model.lm_head

            def _forward_for_compile(
                input_ids: torch.Tensor,
                positions: torch.Tensor,
            ) -> torch.Tensor:
                hidden = inner(input_ids, positions)
                # ``inner`` returns ``[N, hidden]`` flat; for decode N==B
                # and we sample on every row.
                logits = lm_head(hidden)
                return logits.argmax(dim=-1)

            # Bump Dynamo cache limits.  Jamba has 32 layers; each layer
            # has a different ``self.layer_idx`` int attribute, so Dynamo
            # treats every layer as a fresh compile target.
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

        def _decode_step():
            set_jamba_context(
                is_prefill=False,
                slot_mapping=self._step_slot_mapping,
                context_lens=self._step_context_lens,
                block_tables=self._block_tables,
                max_context_len=max_context_len,
                mamba_metadata=mamba_meta,
            )
            try:
                if self._compiled_decode_step is not None:
                    tok = self._compiled_decode_step(
                        self._step_input_ids, self._step_positions,
                    )
                else:
                    hidden = self.model(
                        self._step_input_ids, self._step_positions,
                    )
                    logits = self.model.compute_logits(hidden)
                    tok = logits.argmax(dim=-1)
                self._step_next_tokens.copy_(tok)
            finally:
                reset_context()

        # Warmup outside the graph stream so allocator state is settled
        # before capture.  vLLM does the same trick.
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _decode_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _decode_step()

        return _JambaDecodeGraph(
            graph=graph,
            step_input_ids=self._step_input_ids,
            step_positions=self._step_positions,
            slot_mapping=self._step_slot_mapping,
            context_lens=self._step_context_lens,
            next_tokens=self._step_next_tokens,
        )

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
        """Batched generate.  Splits prompts into micro-batches of
        ``max_num_seqs`` and runs each through prefill + decode loop.
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

    # ------------------------------------------------------------------
    # _run_batch: prefill (flat varlen) + decode loop (single-step graph).
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _run_batch(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: list[SamplingParams],
    ) -> list[GenerationOutput]:
        B = len(prompt_token_ids)
        device = self.device
        eos = self.tokenizer.eos_token_id

        max_out = max(p.max_tokens for p in sampling_params)
        prompt_lens = [len(p) for p in prompt_token_ids]
        max_prompt = max(prompt_lens)
        max_total = max_prompt + max_out

        all_greedy = all(p.temperature == 0.0 for p in sampling_params)
        fits_static = max_total <= self.graph_max_total
        use_graph = self._use_cuda_graphs and all_greedy and fits_static

        # Pad to ``max_num_seqs`` by cloning the first real prompt.  Each
        # padded row gets its own paged-KV blocks and Mamba slot via the
        # static block_tables / cache_indices, runs alongside real rows,
        # and its output is discarded.  Cloning instead of dummy short
        # prompts keeps cache_seqlens uniform across the lockstep batch
        # (TRTLLM kernels misbehave under wide cache_seqlens variance).
        B_pad = self.max_num_seqs
        if B < B_pad:
            extra = B_pad - B
            prompt_token_ids = list(prompt_token_ids) + [prompt_token_ids[0]] * extra
            prompt_lens = prompt_lens + [prompt_lens[0]] * extra

        # ------------------------------------------------------------------
        # Reset cache state for the new batch.  Only zero the Mamba state
        # (paged KV slots are overwritten by the prefill side-write so
        # don't need clearing).
        # ------------------------------------------------------------------
        for cs in self._conv_states:
            cs.zero_()
        for ss in self._ssm_states:
            ss.zero_()

        # ------------------------------------------------------------------
        # Prefill: pack flat varlen, run model with standard
        # ``set_jamba_context`` (paged-attn + Mamba metadata in one).
        # ------------------------------------------------------------------
        first_tok = self._run_prefill(
            prompt_token_ids, prompt_lens, B_pad, max_prompt,
        )

        # ------------------------------------------------------------------
        # Decode loop (greedy + graph fast-path).
        # ------------------------------------------------------------------
        if use_graph:
            generated = self._run_decode_graph(
                first_tok, prompt_lens, max_out, sampling_params, B,
            )
        else:
            generated = self._run_decode_eager(
                first_tok, prompt_lens, max_out, sampling_params, B,
            )

        return self._materialise(generated)

    # ------------------------------------------------------------------
    # Prefill: flat-varlen forward.  Returns ``first_tok`` of shape [B_pad].
    # ------------------------------------------------------------------
    def _run_prefill(
        self,
        prompt_token_ids: list[list[int]],
        prompt_lens: list[int],
        B_pad: int,
        max_prompt: int,
    ) -> torch.Tensor:
        device = self.device
        page_size = self._page_size

        # Flat input_ids: concatenate all prompts.
        flat_ids: list[int] = []
        for p in prompt_token_ids:
            flat_ids.extend(p)
        input_ids = torch.tensor(flat_ids, dtype=torch.long, device=device)

        # cu_seqlens (Q == K for prefill).  Cumulative real-token counts.
        plens_np = np.array(prompt_lens, dtype=np.int32)
        cu_q_np = np.zeros(B_pad + 1, dtype=np.int32)
        np.cumsum(plens_np, out=cu_q_np[1:])
        cu_q = torch.from_numpy(cu_q_np).to(device)

        # Per-token positions (real position within each seq, 0..plen-1).
        pos_np = np.concatenate([
            np.arange(p, dtype=np.int64) for p in prompt_lens
        ])
        positions = torch.from_numpy(pos_np).to(device)

        # Per-token slot_mapping (vectorized).  For the k-th real token
        # (in the concatenated flat layout):
        #   seq_idx[k]      = which seq
        #   within_seq[k]   = position 0..plen_seq-1 within that seq
        #   slot[k]         = block_tables[seq_idx[k], within_seq[k]//P] * P
        #                     + (within_seq[k] % P)
        # The Python double-loop over 32*1024=32K tokens used to take
        # ~50ms per prefill on tiny-dev (dominating the prefill-heavy
        # scenario at 1024 prompt tokens); the numpy vector form
        # takes ~0.5ms.
        bt_host = self._block_tables.cpu().numpy()              # [B_pad, bps]
        seq_idx = np.repeat(
            np.arange(len(prompt_lens), dtype=np.int64), plens_np,
        )                                                       # [N_real]
        within_seq = pos_np                                     # [N_real]
        block_idxs = within_seq // page_size                    # [N_real]
        slot_in_blocks = within_seq % page_size                 # [N_real]
        slot_np = (
            bt_host[seq_idx, block_idxs].astype(np.int64) * page_size
            + slot_in_blocks
        )
        slot_mapping = torch.from_numpy(slot_np).to(device)

        # Mamba prefill metadata: query_start_loc + has_initial_state.
        mamba_qsl = cu_q  # same as cu_seqlens for prefill
        mamba_has_init = torch.zeros(B_pad, dtype=torch.bool, device=device)

        mamba_meta = JambaMambaMetadata(
            conv_states=self._conv_states,
            ssm_states=self._ssm_states,
            cache_indices=self._cache_indices,
            is_decode=False,
            query_start_loc=mamba_qsl,
            has_initial_state=mamba_has_init,
        )

        max_seqlen = int(plens_np.max())
        set_jamba_context(
            is_prefill=True,
            slot_mapping=slot_mapping,
            block_tables=self._block_tables,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_q,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            mamba_metadata=mamba_meta,
        )
        try:
            hidden = self.model(input_ids, positions)
            # Hidden is ``[total_real_tokens, hidden]`` -- gather the
            # last token of each seq for the LM head.  cu_q[1:] - 1 are
            # the per-seq last-token indices.
            last_idx = cu_q[1:].long() - 1
            last_hidden = hidden.index_select(0, last_idx)        # [B_pad, hidden]
            prefill_logits = self.model.compute_logits(last_hidden)
        finally:
            reset_context()
        return prefill_logits.argmax(dim=-1)                     # [B_pad]

    # ------------------------------------------------------------------
    # Decode: graph fast-path.  Greedy, lockstep, fixed B=max_num_seqs.
    # ------------------------------------------------------------------
    def _run_decode_graph(
        self,
        first_tok: torch.Tensor,
        prompt_lens: list[int],
        max_out: int,
        sampling_params: list[SamplingParams],
        B_real: int,
    ) -> list[list[int]]:
        device = self.device
        eos = self.tokenizer.eos_token_id
        B_pad = self.max_num_seqs
        page_size = self._page_size
        graph_entry = self._decode_graph
        assert graph_entry is not None

        # Pre-compute the per-step slot_mapping / context_lens / positions
        # tables on host (vectorized -- the python loop took ~5ms/batch
        # at max_out=128, B_pad=32 prior).
        bt_host = self._block_tables.cpu().numpy()
        prompt_lens_np = np.array(prompt_lens, dtype=np.int64)

        s_arr = np.arange(max_out, dtype=np.int64)[:, None]       # [max_out, 1]
        positions_table = prompt_lens_np[None, :] + s_arr         # [max_out, B_pad]
        block_idxs = positions_table // page_size
        slot_in_blocks = positions_table % page_size
        row_idx = np.arange(B_pad, dtype=np.int64)[None, :]
        slot_table = (
            bt_host[row_idx, block_idxs].astype(np.int64) * page_size
            + slot_in_blocks
        )
        ctxlen_table = (positions_table + 1).astype(np.int32)
        slot_table_t = torch.from_numpy(slot_table).to(device)
        ctxlen_table_t = torch.from_numpy(ctxlen_table).to(device)
        positions_table_t = torch.from_numpy(positions_table).to(device)

        # Token history buffer: [max_out, B_pad] int64.  pin_memory + non
        # blocking copies make the host->device round-trip overlap with
        # the next graph replay.
        tok_history = torch.empty(
            max_out, B_pad, dtype=torch.long, pin_memory=True,
        )
        tok_history[0].copy_(first_tok, non_blocking=True)

        # Step 0 input: first_tok (sampled from prefill logits).
        self._step_input_ids.copy_(first_tok)

        for step in range(1, max_out):
            # Update static buffers in-place: the captured graph reads
            # from these identities each replay.
            self._step_slot_mapping.copy_(slot_table_t[step - 1])
            self._step_context_lens.copy_(ctxlen_table_t[step - 1])
            self._step_positions.copy_(positions_table_t[step - 1])
            graph_entry.graph.replay()
            tok_history[step].copy_(self._step_next_tokens, non_blocking=True)
            self._step_input_ids.copy_(self._step_next_tokens)

        torch.cuda.synchronize()

        # Build per-row generated lists, dropping padding rows.
        history_t = tok_history.numpy()
        generated: list[list[int]] = [[] for _ in range(B_real)]
        for i in range(B_real):
            limit = sampling_params[i].max_tokens
            tokens_i: list[int] = []
            for s in range(min(max_out, limit)):
                t = int(history_t[s, i])
                tokens_i.append(t)
                if (not sampling_params[i].ignore_eos
                        and eos is not None and t == eos):
                    break
            generated[i] = tokens_i
        return generated

    # ------------------------------------------------------------------
    # Decode: eager fallback (graph disabled or sampling != greedy).
    # ------------------------------------------------------------------
    def _run_decode_eager(
        self,
        first_tok: torch.Tensor,
        prompt_lens: list[int],
        max_out: int,
        sampling_params: list[SamplingParams],
        B_real: int,
    ) -> list[list[int]]:
        device = self.device
        eos = self.tokenizer.eos_token_id
        B_pad = self.max_num_seqs
        page_size = self._page_size
        max_context_len = self._blocks_per_seq * self._page_size

        bt_host = self._block_tables.cpu().numpy()
        prompt_lens_np = np.array(prompt_lens, dtype=np.int64)

        generated: list[list[int]] = [[] for _ in range(B_pad)]
        for i in range(B_pad):
            generated[i].append(int(first_tok[i].item()))
        finished = [False] * B_pad
        for i in range(B_real):
            if (not sampling_params[i].ignore_eos
                    and eos is not None and generated[i][0] == eos):
                finished[i] = True
            if len(generated[i]) >= sampling_params[i].max_tokens:
                finished[i] = True

        step_input_ids = first_tok.clone()
        slot_mapping = torch.empty(B_pad, dtype=torch.long, device=device)
        context_lens = torch.empty(B_pad, dtype=torch.int32, device=device)
        positions = torch.empty(B_pad, dtype=torch.long, device=device)

        for step in range(1, max_out):
            if all(finished[:B_real]):
                break
            slots_h = np.empty(B_pad, dtype=np.int64)
            ctx_h = np.empty(B_pad, dtype=np.int32)
            pos_h = np.empty(B_pad, dtype=np.int64)
            for i in range(B_pad):
                pos = int(prompt_lens_np[i]) + step - 1
                block_idx = pos // page_size
                slot_in_block = pos % page_size
                slots_h[i] = int(bt_host[i, block_idx]) * page_size + slot_in_block
                ctx_h[i] = pos + 1
                pos_h[i] = pos
            slot_mapping.copy_(torch.from_numpy(slots_h))
            context_lens.copy_(torch.from_numpy(ctx_h))
            positions.copy_(torch.from_numpy(pos_h))

            mamba_meta = JambaMambaMetadata(
                conv_states=self._conv_states,
                ssm_states=self._ssm_states,
                cache_indices=self._cache_indices,
                is_decode=True,
            )
            set_jamba_context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=self._block_tables,
                max_context_len=max_context_len,
                mamba_metadata=mamba_meta,
            )
            try:
                hidden = self.model(step_input_ids, positions)
                logits = self.model.compute_logits(hidden)
            finally:
                reset_context()
            next_tokens = self._sample_step(logits, sampling_params[:B_real]
                                            + [sampling_params[0]] * (B_pad - B_real))
            step_input_ids = torch.tensor(
                next_tokens, dtype=torch.long, device=device,
            )
            for i in range(B_pad):
                if i < B_real and finished[i]:
                    continue
                t = next_tokens[i]
                generated[i].append(int(t))
                if i < B_real:
                    sp = sampling_params[i]
                    if not sp.ignore_eos and eos is not None and t == eos:
                        finished[i] = True
                    if len(generated[i]) >= sp.max_tokens:
                        finished[i] = True

        return generated[:B_real]

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
