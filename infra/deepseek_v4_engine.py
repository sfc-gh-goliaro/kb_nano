"""Dedicated serving engine for DeepSeek-V4-Flash (kb_nano / fastkernels).

DeepSeek V4's attention stack (sparse sliding-window MLA with per-layer
compression ratios {1,4,128}, attention sink, ``fp8_ds_mla`` paged cache,
Lightning indexer + compressor) and MXFP4 routed experts are inseparable from
vLLM's V1 GPU runner: the attention modules read their per-layer metadata from
vLLM's forward context (``DeepseekSparseSWAMetadata`` / ``FlashMLASparseMetadata``
/ indexer metadata), which is produced by vLLM's attention-metadata builders and
KV-cache-group machinery.

Per the chosen strategy (reuse the exact compiled kernels from the installed
vLLM 0.20.0 wheel), this engine runs DeepSeek V4 by driving vLLM's V1 execution
core, while exposing the kb_nano ``LlamaEngine`` serving interface the
benchmark harness expects (``generate`` over token-id prompts, ``block_manager
.reset()``, kb_nano ``SamplingParams`` / ``GenerationOutput``). This is the
same "dedicated per-pipeline engine" pattern kb_nano already uses for models
that do not fit the generic paged-KV path (jamba/pi0/diffusion).

Fast iteration knobs (env):
  FASTKERNELS_V4_NUM_LAYERS=N   build only the first N decoder layers
  FASTKERNELS_V4_DUMMY=1        random (dummy) weights, skip the 160GB read
"""

from __future__ import annotations

import os

from .engine import GenerationOutput, SamplingParams


class _BlockManagerShim:
    """kb_nano benchmark calls ``engine.block_manager.reset()`` between
    scenarios.  vLLM resets its own KV-cache/prefix state per ``generate``
    call (prefix caching is disabled here), so this is a no-op shim that keeps
    the harness interface intact."""

    def reset(self) -> None:
        return None


class DeepseekV4Engine:
    def __init__(
        self,
        model_name: str,
        seed: int = 0,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        **_ignored,
    ):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM

        llm_kwargs = dict(
            model=model_name,
            seed=seed,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
            # DeepSeek V4 only supports the fp8 (fp8_ds_mla) KV-cache layout.
            kv_cache_dtype="fp8",
        )

        # --- Fast-iteration overrides -------------------------------------
        num_layers = os.environ.get("FASTKERNELS_V4_NUM_LAYERS")
        if num_layers:
            llm_kwargs["hf_overrides"] = {"num_hidden_layers": int(num_layers)}
        if os.environ.get("FASTKERNELS_V4_DUMMY", "0") == "1":
            llm_kwargs["load_format"] = "dummy"
        elif os.environ.get("FASTKERNELS_V4_LOAD_FORMAT"):
            llm_kwargs["load_format"] = os.environ["FASTKERNELS_V4_LOAD_FORMAT"]

        self.llm = LLM(**llm_kwargs)
        self.block_manager = _BlockManagerShim()
        self.model_name = model_name

    def _to_vllm_sp(self, sp: SamplingParams):
        from vllm import SamplingParams as VLLMSamplingParams

        return VLLMSamplingParams(
            temperature=sp.temperature,
            top_p=sp.top_p,
            max_tokens=sp.max_tokens,
            seed=sp.seed,
            ignore_eos=sp.ignore_eos,
            detokenize=True,
        )

    def generate(
        self,
        prompts,
        sampling_params,
        use_tqdm: bool = False,
        decode_text: bool = True,
        **_ignored,
    ):
        """``prompts`` is a list of token-id lists (decode_text=False) or
        strings.  ``sampling_params`` is a kb_nano ``SamplingParams`` or a
        list of them (one per prompt), matching the benchmark worker."""
        from vllm import SamplingParams as VLLMSamplingParams  # noqa: F401

        if isinstance(sampling_params, (list, tuple)):
            sp_list = [self._to_vllm_sp(sp) for sp in sampling_params]
        else:
            sp_list = self._to_vllm_sp(sampling_params)

        vllm_prompts = []
        for p in prompts:
            if isinstance(p, str):
                vllm_prompts.append({"prompt": p})
            else:
                vllm_prompts.append({"prompt_token_ids": list(p)})

        outputs = self.llm.generate(vllm_prompts, sp_list, use_tqdm=use_tqdm)

        results = []
        for o in outputs:
            comp = o.outputs[0]
            results.append(
                GenerationOutput(
                    prompt="",
                    generated_text=comp.text if decode_text else "",
                    token_ids=list(comp.token_ids),
                )
            )
        return results
