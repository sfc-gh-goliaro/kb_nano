#!/usr/bin/env python3
"""LLM-powered kernel generation agent.

Uses Claude Opus 4.6 (via the internal Corvo endpoint) to generate
replacement kernels for kb_nano operators, then benchmarks them using
the kb_nano.bench suite.

Usage:
    python -m kb_nano.example.agent \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --level 1

    python -m kb_nano.example.agent \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --level 1 --cuda-only --max-retries 3

    python -m kb_nano.example.agent \
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --level 2 --tp 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from kb_nano.example.llm_api import call_llm

_OUTPUT_DIR = Path(__file__).resolve().parent / "_generated_kernels"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class OperatorSpec:
    """Info about a single benchmark target operator."""
    name: str
    level: int
    module_path: str
    source_code: str
    class_name: str
    models: list[str]


@dataclass
class GeneratedKernel:
    """A generated replacement kernel for one operator."""
    op_name: str
    class_name: str
    code: str
    file_path: str | None = None
    error: str | None = None
    success: bool = False
    attempts: int = 0


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def _detect_model_key(model_name: str) -> str:
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(hf_config, "model_type", "llama")
    key_map = {"llama": "llama31", "mixtral": "mixtral"}
    key = key_map.get(model_type)
    if key is None:
        raise ValueError(f"Unsupported model type {model_type!r} for {model_name}")
    return key


def discover_operators(model_name: str, level: int) -> list[OperatorSpec]:
    """Find all operators at the given level used by the given model."""
    from kb_nano.bench.discovery import discover_targets

    model_key = _detect_model_key(model_name)
    targets = discover_targets()

    kb_root = Path(__file__).resolve().parent.parent

    ops = []
    for t in targets:
        if t.level != level:
            continue
        if model_key not in t.models:
            continue

        mod_file = kb_root / t.module_path.replace(".", "/")
        mod_file = mod_file.with_suffix(".py")
        try:
            source = mod_file.read_text()
        except OSError:
            continue

        ops.append(OperatorSpec(
            name=t.name,
            level=t.level,
            module_path=t.module_path,
            source_code=source,
            class_name=t.target_cls.__name__,
            models=t.models,
        ))

    return ops


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer. You write high-performance CUDA, "
    "Triton, and PyTorch kernels for LLM inference. Your code is always correct, "
    "compilable, and production-quality."
)


def build_generation_prompt(op: OperatorSpec, cuda_only: bool) -> str:
    constraint_block = _constraint_text(cuda_only)
    return (
        "I need you to write a high-performance replacement for the following "
        "PyTorch nn.Module operator used in an LLM inference engine. "
        "Your goal is to produce a kernel that is FASTER than the baseline "
        "implementation while remaining numerically correct.\n\n"
        "## Baseline implementation\n\n"
        f"```python\n{op.source_code}\n```\n\n"
        "## Requirements\n\n"
        f"1. Your replacement class MUST:\n"
        f"   - Be named `{op.class_name}` (exactly)\n"
        f"   - Subclass `torch.nn.Module`\n"
        f"   - Have the EXACT same `forward` signature (same parameter names, types, defaults)\n"
        f"   - Produce numerically equivalent outputs (or very close)\n"
        f"   - The `__init__` method is optional -- you only need to override it if you need "
        f"to change initialization logic (e.g. pre-allocate buffers). If you do override it, "
        f"keep the same signature.\n"
        f"2. {constraint_block}\n"
        f"3. Do NOT import `vllm`, `sglang`, or `sgl_kernel`.\n"
        f"4. You may import `torch`, `triton`, `triton.language`, standard library modules, "
        f"`flash_attn`, or JIT-compile CUDA. For inline CUDA strings use "
        f"`torch.utils.cpp_extension.load_inline(name=..., cpp_sources=..., "
        f"cuda_sources=..., functions=[...])`. Do NOT pass `cuda_sources` to "
        f"`torch.utils.cpp_extension.load()` (it only takes file paths via `sources`).\n"
        f"5. Focus on PERFORMANCE: minimize memory traffic, maximize GPU occupancy, "
        f"fuse operations where possible, and use vectorized loads/stores.\n\n"
        "## Response format\n\n"
        "Return ONLY a single Python code block (```python ... ```) containing:\n"
        "- All necessary imports at the top\n"
        f"- The class definition for `{op.class_name}`\n"
        "- Any helper functions, Triton kernels, or CUDA source strings needed\n\n"
        "Do NOT include any explanation outside the code block. Do NOT include "
        "if __name__ == '__main__' blocks or test code."
    )


def _constraint_text(cuda_only: bool) -> str:
    if cuda_only:
        return (
            "You MUST use raw CUDA kernels for the core computation. "
            "Do NOT use Triton kernels, PyTorch built-in functions "
            "(F.linear, F.embedding, etc.), flash_attn, torch.distributed, "
            "or external Python libraries for the core computation. "
            "You may use CUDA libraries like cuBLAS, cutlass, cuDNN, nccl, etc.\n"
            "   IMPORTANT: To JIT-compile inline CUDA source strings, use "
            "`torch.utils.cpp_extension.load_inline(name=..., cpp_sources=..., "
            "cuda_sources=..., functions=[...])`. Do NOT pass `cuda_sources` "
            "to `torch.utils.cpp_extension.load()` -- that function only accepts "
            "file paths via `sources=[...]`."
        )
    return (
        "You may use Triton, PyTorch, raw CUDA, or any combination. "
        "Aim for the highest performance possible on NVIDIA H200 GPUs."
    )


def build_retry_prompt(
    op: OperatorSpec, failed_code: str, error_msg: str, cuda_only: bool,
) -> str:
    constraint_block = _constraint_text(cuda_only)
    return (
        f"The kernel you generated for `{op.class_name}` failed with this error:\n\n"
        f"```\n{error_msg}\n```\n\n"
        f"Here is the code that failed:\n\n"
        f"```python\n{failed_code}\n```\n\n"
        f"Here is the original baseline for reference:\n\n"
        f"```python\n{op.source_code}\n```\n\n"
        f"## Requirements (same as before)\n\n"
        f"1. Class must be named `{op.class_name}`, subclass `torch.nn.Module`, "
        f"same `forward` signature (override `__init__` only if needed, keeping the same signature).\n"
        f"2. {constraint_block}\n"
        f"3. Do NOT import `vllm`, `sglang`, or `sgl_kernel`.\n\n"
        f"Please fix the error and return ONLY a corrected Python code block "
        f"(```python ... ```). No explanation outside the code block."
    )


# ---------------------------------------------------------------------------
# Code extraction & validation
# ---------------------------------------------------------------------------
def extract_python_code(response: str) -> str | None:
    """Extract the first Python code block from an LLM response."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def validate_kernel(code: str, expected_class_name: str) -> tuple[type | None, str | None]:
    """Write code to a temp file, import it, instantiate the class, and check it works.

    Returns (cls, None) on success, (None, error_msg) on failure.
    """
    tmp_dir = tempfile.mkdtemp(prefix="kb_kernel_")
    tmp_file = os.path.join(tmp_dir, "generated_kernel.py")

    with open(tmp_file, "w") as f:
        f.write(code)

    try:
        spec = importlib.util.spec_from_file_location("_gen_kernel", tmp_file)
        if spec is None or spec.loader is None:
            return None, f"Cannot create module spec from {tmp_file}"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cls = getattr(mod, expected_class_name, None)
        if cls is None:
            available = [
                n for n, v in vars(mod).items()
                if isinstance(v, type) and issubclass(v, __import__("torch").nn.Module)
            ]
            return None, (
                f"Class {expected_class_name!r} not found in generated code. "
                f"Available nn.Module classes: {available}"
            )

        # Try instantiating to catch deferred JIT compilation errors
        try:
            cls()
        except Exception:
            return None, (
                f"Class {expected_class_name} found but __init__() failed:\n"
                + traceback.format_exc()
            )

        return cls, None

    except Exception:
        return None, traceback.format_exc()


# ---------------------------------------------------------------------------
# Kernel generation loop
# ---------------------------------------------------------------------------
def _save_kernel(op_name: str, code: str) -> Path:
    """Write kernel code to _generated_kernels/<op_name>.py and return the path."""
    out_file = _OUTPUT_DIR / f"{op_name}.py"
    out_file.write_text(code)
    return out_file


def generate_kernel(
    op: OperatorSpec,
    cuda_only: bool,
    max_retries: int,
    llm_model: str,
) -> GeneratedKernel:
    """Generate a replacement kernel for one operator, with retries."""
    result = GeneratedKernel(op_name=op.name, class_name=op.class_name, code="")

    prompt = build_generation_prompt(op, cuda_only)
    print(f"\n  Generating {op.class_name} ({op.name})...")

    for attempt in range(1, max_retries + 1):
        result.attempts = attempt
        print(f"    Attempt {attempt}/{max_retries}... ", end="", flush=True)

        try:
            response = call_llm(
                prompt, model_name=llm_model, max_tokens=8192, temperature=0.0,
                system=SYSTEM_PROMPT,
            )
        except Exception as e:
            error_msg = f"LLM API call failed: {e}"
            print(f"API error: {e}")
            result.error = error_msg
            if attempt < max_retries:
                time.sleep(2)
            continue

        code = extract_python_code(response)
        if code is None:
            error_msg = "No Python code block found in LLM response."
            print("no code block found")
            result.error = error_msg
            result.code = response[:500]
            prompt = build_retry_prompt(
                op, response[:500], error_msg, cuda_only,
            )
            continue

        result.code = code
        out_file = _save_kernel(op.name, code)
        result.file_path = str(out_file)

        cls, error_msg = validate_kernel(code, op.class_name)
        if cls is not None:
            print("OK")
            result.success = True
            result.error = None
            return result

        print(f"validation failed")
        result.error = error_msg
        print(f"      Error: {_truncate(error_msg, 200)}")

        if attempt < max_retries:
            prompt = build_retry_prompt(op, code, error_msg, cuda_only)

    print(f"    FAILED after {max_retries} attempts")
    return result


def _regenerate_kernel(
    op: OperatorSpec,
    old_kernel: GeneratedKernel,
    runtime_error: str,
    cuda_only: bool,
    max_retries: int,
    llm_model: str,
) -> GeneratedKernel:
    """Re-generate a kernel that failed at runtime, feeding the error to the LLM."""
    result = GeneratedKernel(
        op_name=op.name, class_name=op.class_name, code=old_kernel.code,
        file_path=old_kernel.file_path,
    )
    prompt = build_retry_prompt(op, old_kernel.code, runtime_error, cuda_only)
    print(f"\n  Re-generating {op.class_name} ({op.name}) after runtime error...")

    for attempt in range(1, max_retries + 1):
        result.attempts = attempt
        print(f"    Attempt {attempt}/{max_retries}... ", end="", flush=True)

        try:
            response = call_llm(
                prompt, model_name=llm_model, max_tokens=8192, temperature=0.0,
                system=SYSTEM_PROMPT,
            )
        except Exception as e:
            print(f"API error: {e}")
            result.error = f"LLM API call failed: {e}"
            if attempt < max_retries:
                time.sleep(2)
            continue

        code = extract_python_code(response)
        if code is None:
            print("no code block found")
            result.error = "No Python code block found in LLM response."
            result.code = response[:500]
            prompt = build_retry_prompt(op, response[:500], result.error, cuda_only)
            continue

        result.code = code
        out_file = _save_kernel(op.name, code)
        result.file_path = str(out_file)

        cls, error_msg = validate_kernel(code, op.class_name)
        if cls is not None:
            print("OK")
            result.success = True
            result.error = None
            return result

        print("validation failed")
        result.error = error_msg
        print(f"      Error: {_truncate(error_msg, 200)}")

        if attempt < max_retries:
            prompt = build_retry_prompt(op, code, error_msg, cuda_only)

    print(f"    FAILED after {max_retries} attempts")
    return result


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len] + "..."


# ---------------------------------------------------------------------------
# Benchmark subprocess
# ---------------------------------------------------------------------------
_BENCH_WORKER = r'''
import json, os, sys, time, gc, traceback

def main():
    cfg = json.loads(sys.argv[1])
    sys.path.insert(0, cfg["project_root"])

    import torch
    from importlib.util import spec_from_file_location, module_from_spec

    pkg = cfg["package_name"]
    bench_discovery = __import__(f"{pkg}.bench.discovery", fromlist=["discover_targets", "get"])
    bench_replacement = __import__(f"{pkg}.bench.replacement", fromlist=["patch_class", "restore"])
    bench_evaluator = __import__(f"{pkg}.bench.evaluator", fromlist=["evaluate"])
    engine_mod = __import__(f"{pkg}.engine", fromlist=["LlamaEngine", "SamplingParams"])

    LlamaEngine = engine_mod.LlamaEngine
    SamplingParams = engine_mod.SamplingParams

    kernels = cfg["kernels"]  # list of {op_name, class_name, code_file}
    prompts = cfg["prompts"]
    model_name = cfg["model"]
    tp = cfg["tp"]
    seed = cfg["seed"]
    max_tokens = cfg["max_tokens"]

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)
    engine_kwargs = dict(
        model_name=model_name, seed=seed,
        enforce_eager=True, tensor_parallel_size=tp,
    )

    # --- Baseline run ---
    print("  [bench] Building baseline model...")
    engine = LlamaEngine(**engine_kwargs)
    try:
        engine.generate(["warmup"], sp, collect_logits=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        baseline_outputs = engine.generate(prompts, sp, collect_logits=True)
        torch.cuda.synchronize()
        baseline_time = time.perf_counter() - t0
    finally:
        engine._cleanup()
        del engine
    gc.collect()
    torch.cuda.empty_cache()

    # --- Patch all operators ---
    undo_list = []
    patched_names = []
    for kern in kernels:
        try:
            target = bench_discovery.get(kern["op_name"])
            spec = spec_from_file_location("_gen_kernel_" + kern["op_name"], kern["code_file"])
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            user_cls = getattr(mod, kern["class_name"])
            undo = bench_replacement.patch_class(target, user_cls)
            undo_list.extend(undo)
            patched_names.append(kern["op_name"])
            print(f"  [bench] Patched {kern['op_name']} -> {kern['class_name']}")
        except Exception as e:
            print(f"  [bench] WARNING: Failed to patch {kern['op_name']}: {e}")
            traceback.print_exc()

    # --- User run ---
    results_out = {}
    try:
        print("  [bench] Building model with user kernels...")
        engine = LlamaEngine(**engine_kwargs)
        try:
            engine.generate(["warmup"], sp, collect_logits=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            user_outputs = engine.generate(prompts, sp, collect_logits=True)
            torch.cuda.synchronize()
            user_time = time.perf_counter() - t0
        finally:
            engine._cleanup()
            del engine

        result = bench_evaluator.evaluate(
            "all_patched", model_name,
            baseline_outputs, user_outputs,
            baseline_time, user_time,
        )
        results_out = {
            "kl_mean": result.kl_mean,
            "kl_max": result.kl_max,
            "token_match_rate": result.token_match_rate,
            "num_tokens": result.num_tokens,
            "baseline_time": result.baseline_time,
            "user_time": result.user_time,
            "speedup": result.speedup,
            "patched_ops": patched_names,
            "success": True,
        }
    except Exception as e:
        results_out = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "patched_ops": patched_names,
        }
    finally:
        bench_replacement.restore(undo_list)

    with open(cfg["output_file"], "w") as f:
        json.dump(results_out, f)

if __name__ == "__main__":
    main()
'''


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


def _identify_failing_kernel(
    error_info: dict, kernels: list[GeneratedKernel],
) -> GeneratedKernel | None:
    """Parse a benchmark error to find which generated kernel caused it."""
    tb = error_info.get("traceback", "") + "\n" + error_info.get("error", "")
    for k in kernels:
        if not k.file_path:
            continue
        if k.file_path in tb or f"_generated_kernels/{k.op_name}.py" in tb:
            return k
    return None


def run_benchmark(
    kernels: list[GeneratedKernel],
    model_name: str,
    tp: int,
    max_tokens: int,
    seed: int,
) -> dict:
    """Run the benchmark in a subprocess with all generated kernels patched."""
    successful = [k for k in kernels if k.success and k.file_path]
    if not successful:
        return {"success": False, "error": "No kernels compiled successfully"}

    kernel_specs = []
    for k in successful:
        kernel_specs.append({
            "op_name": k.op_name,
            "class_name": k.class_name,
            "code_file": k.file_path,
        })

    tmp_dir = tempfile.mkdtemp(prefix="kb_bench_")

    worker_file = os.path.join(tmp_dir, "_worker.py")
    with open(worker_file, "w") as f:
        f.write(_BENCH_WORKER)

    output_file = os.path.join(tmp_dir, "_results.json")

    pkg_dir = Path(__file__).resolve().parent.parent
    project_root = str(pkg_dir.parent)
    package_name = pkg_dir.name

    config = {
        "project_root": project_root,
        "package_name": package_name,
        "model": model_name,
        "tp": tp,
        "seed": seed,
        "max_tokens": max_tokens,
        "prompts": DEFAULT_PROMPTS,
        "kernels": kernel_specs,
        "output_file": output_file,
    }

    config_str = json.dumps(config)

    print(f"\n{'=' * 70}")
    print("  Running benchmark subprocess...")
    print(f"  Model: {model_name}  TP={tp}  Operators: {[k.op_name for k in successful]}")
    print(f"{'=' * 70}")

    result = subprocess.run(
        [sys.executable, worker_file, config_str],
        capture_output=True, text=True,
        timeout=600,
    )

    # Always relay subprocess stdout so the user can see progress
    if result.stdout:
        print(result.stdout, end="")

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-3000:]
        return {
            "success": False,
            "error": f"Benchmark subprocess exited with code {result.returncode}",
            "traceback": stderr_tail,
        }

    try:
        with open(output_file) as f:
            return json.loads(f.read())
    except Exception as e:
        return {"success": False, "error": f"Failed to read results: {e}"}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_generation_report(kernels: list[GeneratedKernel]) -> None:
    print(f"\n{'=' * 70}")
    print("  KERNEL GENERATION SUMMARY")
    print(f"{'=' * 70}")

    for k in kernels:
        status = "OK" if k.success else "FAILED"
        print(f"  {k.op_name:<25} {status:>8}  (attempts: {k.attempts})")
        if k.error:
            print(f"    Last error: {_truncate(k.error, 120)}")

    ok = sum(1 for k in kernels if k.success)
    print(f"\n  {ok}/{len(kernels)} kernels compiled successfully")


def print_benchmark_report(bench_result: dict) -> None:
    print(f"\n{'=' * 70}")
    print("  BENCHMARK RESULTS")
    print(f"{'=' * 70}")

    if not bench_result.get("success"):
        print(f"  FAILED: {bench_result.get('error', 'unknown error')}")
        tb = bench_result.get("traceback")
        if tb:
            print(f"\n{tb}")
        return

    print(f"  Patched operators: {bench_result['patched_ops']}")
    print(f"  KL divergence:     mean={bench_result['kl_mean']:.6f}  "
          f"max={bench_result['kl_max']:.6f}")
    print(f"  Token match rate:  {bench_result['token_match_rate']:.1%} "
          f"({bench_result['num_tokens']} tokens)")
    print(f"  Baseline time:     {bench_result['baseline_time']:.3f}s")
    print(f"  User time:         {bench_result['user_time']:.3f}s")
    print(f"  Speedup:           {bench_result['speedup']:.2f}x")

    kl = bench_result["kl_mean"]
    match = bench_result["token_match_rate"]
    speedup = bench_result["speedup"]

    print(f"\n  Verdict: ", end="")
    if kl > 0.1:
        print("POOR correctness (KL > 0.1)")
    elif match < 0.5:
        print("POOR correctness (token match < 50%)")
    elif speedup >= 1.0:
        print(f"GOOD -- {speedup:.2f}x speedup with acceptable correctness")
    else:
        print(f"CORRECT but slower ({speedup:.2f}x)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered kernel generation agent for kb_nano",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--level", type=int, required=True, choices=[1, 2, 3, 4],
        help="Operator level to generate (1=kernels, 2=blocks, 3=decoders, 4=models)",
    )
    parser.add_argument(
        "--cuda-only", action="store_true",
        help="Force generated kernels to use raw CUDA only (no Triton/PyTorch builtins)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="Max retries for kernels that fail to compile (default: 5)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Max tokens per prompt during benchmarking (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--llm-model", type=str, default="claude-opus-4-6",
        help="LLM model to use for kernel generation (default: claude-opus-4-6)",
    )
    args = parser.parse_args()

    # Clear previous generated kernels
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  kb-nano LLM Kernel Generation Agent")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Level:       L{args.level}")
    print(f"  CUDA only:   {args.cuda_only}")
    print(f"  Max retries: {args.max_retries}")
    print(f"  LLM model:   {args.llm_model}")
    print(f"  TP:          {args.tp}")
    print(f"  Output:      {_OUTPUT_DIR}")
    print("=" * 70)

    # Step 1: Discover operators
    print("\n  Discovering operators...")
    ops = discover_operators(args.model, args.level)
    if not ops:
        print(f"  No L{args.level} operators found for {args.model}")
        sys.exit(1)

    print(f"  Found {len(ops)} operators:")
    for op in ops:
        print(f"    L{op.level} {op.name:<25} {op.class_name}")

    # Step 2: Generate kernels
    kernels = []
    for op in ops:
        kernel = generate_kernel(op, args.cuda_only, args.max_retries, args.llm_model)
        kernels.append(kernel)

    print_generation_report(kernels)

    # Step 3: Benchmark (with retry loop for runtime failures)
    # Build a map from op_name -> (OperatorSpec, GeneratedKernel) for retry
    op_by_name = {op.name: op for op in ops}
    bench_attempt = 0
    max_bench_retries = args.max_retries

    while True:
        bench_attempt += 1
        bench_result = run_benchmark(
            kernels, args.model, args.tp, args.max_tokens, args.seed,
        )

        if bench_result.get("success"):
            break

        # Try to identify and fix the failing kernel
        failing = _identify_failing_kernel(bench_result, kernels)
        if failing is None or bench_attempt > max_bench_retries:
            if failing is None:
                print(f"\n  Could not identify which kernel caused the failure.")
            else:
                print(f"\n  Exhausted {max_bench_retries} benchmark retries.")
            break

        error_text = bench_result.get("traceback", bench_result.get("error", ""))
        print(f"\n  Benchmark failed due to kernel '{failing.op_name}'. "
              f"Re-generating (bench retry {bench_attempt}/{max_bench_retries})...")

        op = op_by_name[failing.op_name]
        # Mark the kernel as failed so we can re-generate
        failing.success = False
        failing.error = error_text

        # Use the existing generate_kernel to re-generate with error context
        # Start with a retry prompt seeded by the runtime error
        new_kernel = _regenerate_kernel(
            op, failing, error_text, args.cuda_only, args.max_retries, args.llm_model,
        )

        # Replace the old kernel in the list
        for i, k in enumerate(kernels):
            if k.op_name == failing.op_name:
                kernels[i] = new_kernel
                break

        if not new_kernel.success:
            print(f"  Could not fix kernel '{failing.op_name}' -- "
                  f"excluding it from the benchmark.")
            continue

    print_benchmark_report(bench_result)


if __name__ == "__main__":
    main()
