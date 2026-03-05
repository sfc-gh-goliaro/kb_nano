"""
Orchestrates baseline vs user-kernel inference runs.

Builds the model twice: once with baseline classes (for reference outputs),
then again after monkey-patching the target class with the user's replacement
(for user outputs). Evaluates correctness and performance.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoConfig

from kb_nano.engine import GenerationOutput, LlamaEngine, SamplingParams
from kb_nano.infra.kernel_swapper import BenchTarget, get, patch_class, restore
from .evaluator import BenchResult, evaluate

_MODEL_TYPE_TO_KEY = {
    "llama": "llama31",
    "mixtral": "mixtral",
}

DEFAULT_PROMPTS = [
    "What is 2 + 2?",
    "Translate 'hello' into French, German, and Japanese.",
    (
        "Explain the difference between a stack and a queue in computer "
        "science. Give a real-world analogy for each."
    ),
    (
        "Write a Python function that computes the factorial of a number "
        "using recursion. Include a docstring."
    ),
]


def _detect_model_key(model_name: str) -> str:
    """Map a HuggingFace model name to the registry model key."""
    hf_config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(hf_config, "model_type", "llama")
    key = _MODEL_TYPE_TO_KEY.get(model_type)
    if key is None:
        raise ValueError(
            f"Unsupported model type {model_type!r} for model {model_name}. "
            f"Supported: {list(_MODEL_TYPE_TO_KEY.keys())}"
        )
    return key


def _applicable_models(target: BenchTarget, requested_models: list[str] | None) -> list[str]:
    """Return the list of HF model names to benchmark against."""
    if requested_models:
        return requested_models
    defaults = {
        "llama31": "meta-llama/Llama-3.1-8B-Instruct",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }
    return [defaults[k] for k in target.models if k in defaults]


def _timed_generate(
    engine: LlamaEngine,
    prompts: list[str],
    sampling_params: SamplingParams,
    collect_logits: bool,
    num_warmup: int = 1,
    num_runs: int = 1,
) -> tuple[list[GenerationOutput], float]:
    """Run inference with warmup, return outputs and average wall-clock time."""
    for _ in range(num_warmup):
        engine.generate(prompts, sampling_params, collect_logits=False)

    total_time = 0.0
    outputs = None
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = engine.generate(prompts, sampling_params, collect_logits=collect_logits)
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

    avg_time = total_time / num_runs
    return outputs, avg_time


def run_benchmark(
    target_name: str,
    user_impl: Any,
    models: list[str] | None = None,
    prompts: list[str] | None = None,
    max_tokens: int = 50,
    tp: int = 1,
    seed: int = 42,
    num_warmup: int = 1,
    num_runs: int = 3,
    enforce_eager: bool = True,
) -> list[BenchResult]:
    """Run the benchmark for a given target and user implementation.

    The user_impl must be an nn.Module subclass of the target's class.
    The benchmark builds the model twice: once with baseline classes, then
    again after patching with the user's class.

    Args:
        target_name: Canonical name from discovery (e.g. "rms_norm", "attention").
        user_impl: nn.Module subclass of the target class.
        models: HuggingFace model names to test. Defaults to all applicable.
        prompts: Prompts for generation. Defaults to built-in set.
        max_tokens: Max tokens per prompt.
        tp: Tensor parallelism degree.
        seed: Random seed for reproducibility.
        num_warmup: Warmup iterations before timing.
        num_runs: Timed iterations to average.
        enforce_eager: Disable CUDA graphs (recommended for benchmarking replacements).

    Returns:
        List of BenchResult, one per model.
    """
    target = get(target_name)
    model_names = _applicable_models(target, models)
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)

    results = []
    for model_name in model_names:
        model_key = _detect_model_key(model_name)
        if model_key not in target.models:
            print(f"  Skipping {model_name}: target {target_name!r} not applicable to {model_key}")
            continue

        print(f"\n{'=' * 70}")
        print(f"  Benchmark: {target_name} on {model_name}")
        print(f"  TP={tp}  max_tokens={max_tokens}  seed={seed}")
        print(f"{'=' * 70}")

        engine_kwargs = dict(
            model_name=model_name,
            seed=seed,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tp,
        )

        print("  Building baseline model...")
        engine = LlamaEngine(**engine_kwargs)
        try:
            print("  Running baseline...")
            baseline_outputs, baseline_time = _timed_generate(
                engine, prompts, sp, collect_logits=True,
                num_warmup=num_warmup, num_runs=num_runs,
            )
        finally:
            engine._cleanup()
            del engine

        print(f"  Patching {target.target_cls.__name__} with {user_impl.__name__}...")
        undo_info = patch_class(target, user_impl)
        try:
            print("  Building model with user implementation...")
            engine = LlamaEngine(**engine_kwargs)
            try:
                print("  Running with user implementation...")
                user_outputs, user_time = _timed_generate(
                    engine, prompts, sp, collect_logits=True,
                    num_warmup=num_warmup, num_runs=num_runs,
                )
            finally:
                engine._cleanup()
                del engine
        finally:
            restore(undo_info)

        result = evaluate(
            target_name, model_name,
            baseline_outputs, user_outputs,
            baseline_time, user_time,
        )
        print(f"\n{result.report()}")
        results.append(result)

    return results
