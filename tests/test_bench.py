#!/usr/bin/env python3
"""
Test suite for the kb-nano bench module.

Part 1: Unit tests (discovery, evaluator, replacement) -- no GPU required.
Part 2: Integration test via subprocess -- runs the full benchmark pipeline
        with an identity replacement (expect KL~0) and a broken replacement
        (expect KL>>0) on Llama-3.1-8B-Instruct.

Usage:
    python tests/test_bench.py                # run all tests
    python tests/test_bench.py --unit-only    # skip the GPU integration test
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
PACKAGE_NAME = os.path.basename(PACKAGE_DIR)

sys.path.insert(0, PROJECT_ROOT)

bench_pkg = __import__(f"{PACKAGE_NAME}.bench", fromlist=[
    "list_targets", "models_for_target", "targets_for_model",
    "print_model_operator_map", "BenchTarget", "BenchResult", "benchmark",
])
discovery_mod = __import__(f"{PACKAGE_NAME}.bench.discovery", fromlist=["get"])
evaluator_mod = __import__(f"{PACKAGE_NAME}.bench.evaluator", fromlist=[
    "compute_kl_divergence", "compute_token_match_rate", "evaluate",
])
replacement_mod = __import__(f"{PACKAGE_NAME}.bench.replacement", fromlist=[
    "patch_class", "restore", "replacement_context",
])
engine_mod = __import__(f"{PACKAGE_NAME}.engine", fromlist=["GenerationOutput"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_pass_count = 0
_fail_count = 0


def check(condition: bool, label: str):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"    PASS  {label}")
    else:
        _fail_count += 1
        print(f"    FAIL  {label}")


# ---------------------------------------------------------------------------
# Part 1a: Discovery
# ---------------------------------------------------------------------------
def test_discovery():
    print(f"\n{'=' * 60}")
    print("  TEST: Discovery")
    print(f"{'=' * 60}")

    targets = bench_pkg.list_targets()
    check(len(targets) > 0, "list_targets() returns non-empty list")

    for t in targets:
        valid = (
            isinstance(t.name, str) and len(t.name) > 0
            and t.level in (1, 2, 3, 4)
            and len(t.models) > 0
            and isinstance(t.target_cls, type)
            and issubclass(t.target_cls, nn.Module)
        )
        if not valid:
            check(False, f"target {t.name!r} has valid fields")
            break
    else:
        check(True, "all targets have valid name, level, models, target_cls")

    l1_targets = bench_pkg.list_targets(level=1)
    check(
        len(l1_targets) > 0 and all(t.level == 1 for t in l1_targets),
        "list_targets(level=1) returns only L1 targets",
    )

    rms = discovery_mod.get("rms_norm")
    check(rms.name == "rms_norm" and rms.level == 1, 'get("rms_norm") returns correct target')

    try:
        discovery_mod.get("nonexistent_op_xyz")
        check(False, 'get("nonexistent") raises KeyError')
    except KeyError:
        check(True, 'get("nonexistent") raises KeyError')

    llama_targets = bench_pkg.targets_for_model("llama31")
    check(
        len(llama_targets) > 0 and all("llama31" in t.models for t in llama_targets),
        'targets_for_model("llama31") all include "llama31"',
    )

    rms_models = bench_pkg.models_for_target("rms_norm")
    check("llama31" in rms_models, 'models_for_target("rms_norm") contains "llama31"')


# ---------------------------------------------------------------------------
# Part 1b: Evaluator (synthetic tensors)
# ---------------------------------------------------------------------------
def test_evaluator():
    print(f"\n{'=' * 60}")
    print("  TEST: Evaluator")
    print(f"{'=' * 60}")

    vocab = 128

    # Identical logits -> KL = 0
    logits_a = [torch.randn(1, vocab) for _ in range(5)]
    logits_b = [t.clone() for t in logits_a]
    kl_mean, kl_max, kl_steps = evaluator_mod.compute_kl_divergence(logits_a, logits_b)
    check(kl_mean == 0.0 and kl_max == 0.0, "identical logits -> KL mean=0, max=0")
    check(len(kl_steps) == 5, "identical logits -> 5 per-step values")

    # Different logits -> KL > 0
    logits_c = [torch.randn(1, vocab) for _ in range(5)]
    logits_d = [torch.randn(1, vocab) for _ in range(5)]
    kl_mean2, kl_max2, _ = evaluator_mod.compute_kl_divergence(logits_c, logits_d)
    check(kl_mean2 > 0.0, f"different logits -> KL mean={kl_mean2:.4f} > 0")
    check(kl_max2 >= kl_mean2, "KL max >= KL mean")

    # Empty lists
    kl_mean3, kl_max3, kl_steps3 = evaluator_mod.compute_kl_divergence([], [])
    check(
        kl_mean3 == 0.0 and kl_max3 == 0.0 and kl_steps3 == [],
        "empty logits -> (0, 0, [])",
    )

    # Token match rate: identical
    rate, n = evaluator_mod.compute_token_match_rate([1, 2, 3, 4], [1, 2, 3, 4])
    check(rate == 1.0 and n == 4, "identical tokens -> match_rate=1.0")

    # Token match rate: all different
    rate2, n2 = evaluator_mod.compute_token_match_rate([1, 2, 3], [4, 5, 6])
    check(rate2 == 0.0 and n2 == 3, "all-different tokens -> match_rate=0.0")

    # Token match rate: partial
    rate3, n3 = evaluator_mod.compute_token_match_rate([1, 2, 3, 4], [1, 2, 9, 9])
    check(rate3 == 0.5 and n3 == 4, "half-matching tokens -> match_rate=0.5")

    # Token match rate: empty
    rate4, n4 = evaluator_mod.compute_token_match_rate([], [])
    check(rate4 == 1.0 and n4 == 0, "empty tokens -> (1.0, 0)")

    # evaluate() builds correct BenchResult
    GO = engine_mod.GenerationOutput
    baseline_out = [
        GO(prompt="p1", generated_text="t1", token_ids=[1, 2, 3],
           logits_history=[torch.randn(1, vocab) for _ in range(3)]),
    ]
    user_out_same = [
        GO(prompt="p1", generated_text="t1", token_ids=[1, 2, 3],
           logits_history=[l.clone() for l in baseline_out[0].logits_history]),
    ]
    result = evaluator_mod.evaluate("test_op", "test_model", baseline_out, user_out_same, 1.0, 0.5)
    check(result.target_name == "test_op", "evaluate result.target_name correct")
    check(result.model_name == "test_model", "evaluate result.model_name correct")
    check(result.kl_mean == 0.0, "evaluate identical -> KL mean=0")
    check(result.token_match_rate == 1.0, "evaluate identical -> match=1.0")
    check(result.speedup == 2.0, "evaluate speedup = baseline/user = 1.0/0.5 = 2.0")
    check(result.num_tokens == 3, "evaluate num_tokens=3")

    report_str = result.report()
    check("test_op" in report_str and "test_model" in report_str, "report() contains target and model")


# ---------------------------------------------------------------------------
# Part 1c: Replacement patching
# ---------------------------------------------------------------------------
def test_replacement():
    print(f"\n{'=' * 60}")
    print("  TEST: Replacement patching")
    print(f"{'=' * 60}")

    target = discovery_mod.get("rms_norm")
    rms_module = importlib.import_module(f"{PACKAGE_NAME}.{target.module_path}")
    original_cls = target.target_cls

    # Also import a module that references RMSNorm via `from .. import`
    decoder_mod = importlib.import_module(f"{PACKAGE_NAME}.tasks.baseline.L3.llama_decoder")

    class FakeRMSNorm(nn.Module):
        pass

    # patch_class swaps the class in the source module
    undo = replacement_mod.patch_class(target, FakeRMSNorm)
    check(
        getattr(rms_module, original_cls.__name__) is FakeRMSNorm,
        "patch_class replaces the class in the source module",
    )

    # patch_class also swaps in modules that imported it
    check(
        getattr(decoder_mod, original_cls.__name__) is FakeRMSNorm,
        "patch_class replaces the class in importing modules too",
    )

    # restore puts it back everywhere
    replacement_mod.restore(undo)
    check(
        getattr(rms_module, original_cls.__name__) is original_cls,
        "restore brings back the original in source module",
    )
    check(
        getattr(decoder_mod, original_cls.__name__) is original_cls,
        "restore brings back the original in importing modules",
    )

    # replacement_context works as a context manager
    with replacement_mod.replacement_context(target, FakeRMSNorm):
        check(
            getattr(rms_module, original_cls.__name__) is FakeRMSNorm,
            "replacement_context patches inside the block",
        )
    check(
        getattr(rms_module, original_cls.__name__) is original_cls,
        "replacement_context restores after exiting the block",
    )


# ---------------------------------------------------------------------------
# Part 2: Integration test (subprocess with GPU)
# ---------------------------------------------------------------------------
IDENTITY_WORKER = r'''
import json, os, sys, gc
cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["project_root"])

def main():
    pkg = cfg["package_name"]
    bench = __import__(f"{pkg}.bench", fromlist=["benchmark"])
    rms_mod = __import__(f"{pkg}.tasks.baseline.L1.rms_norm", fromlist=["RMSNorm"])

    results = bench.benchmark(
        target_name="rms_norm",
        user_impl=rms_mod.RMSNorm,
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        prompts=["What is 2 + 2?"],
        max_tokens=5,
        num_warmup=0,
        num_runs=1,
        enforce_eager=True,
    )
    r = results[0]
    with open(cfg["output_file"], "w") as f:
        json.dump({
            "kl_mean": r.kl_mean, "kl_max": r.kl_max,
            "token_match_rate": r.token_match_rate,
            "num_tokens": r.num_tokens, "speedup": r.speedup,
        }, f)

if __name__ == "__main__":
    main()
'''

BROKEN_WORKER = r'''
import json, os, sys, gc
cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["project_root"])

def main():
    import torch
    import torch.nn as nn

    pkg = cfg["package_name"]
    bench = __import__(f"{pkg}.bench", fromlist=["benchmark"])

    class BrokenRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, residual=None):
            if residual is None:
                return torch.zeros_like(x)
            else:
                return torch.zeros_like(x), residual

    results = bench.benchmark(
        target_name="rms_norm",
        user_impl=BrokenRMSNorm,
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        prompts=["What is 2 + 2?"],
        max_tokens=5,
        num_warmup=0,
        num_runs=1,
        enforce_eager=True,
    )
    r = results[0]
    with open(cfg["output_file"], "w") as f:
        json.dump({
            "kl_mean": r.kl_mean, "kl_max": r.kl_max,
            "token_match_rate": r.token_match_rate,
            "num_tokens": r.num_tokens,
        }, f)

if __name__ == "__main__":
    main()
'''


def _run_integration_worker(worker_script: str, label: str) -> dict | None:
    """Run an integration worker in a subprocess, return parsed JSON output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(worker_script)
        script_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    config = {
        "project_root": PROJECT_ROOT,
        "package_name": PACKAGE_NAME,
        "output_file": output_path,
    }

    try:
        print(f"    [{label}] Launching subprocess...")
        result = subprocess.run(
            [sys.executable, script_path, json.dumps(config)],
            timeout=600,
        )
        if result.returncode != 0:
            check(False, f"[{label}] subprocess exited with code {result.returncode}")
            return None

        with open(output_path) as f:
            return json.loads(f.read())
    finally:
        os.unlink(script_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_integration():
    print(f"\n{'=' * 60}")
    print("  TEST: Integration (subprocess, GPU required)")
    print(f"{'=' * 60}")

    # --- Identity test (separate subprocess) ---
    id_data = _run_integration_worker(IDENTITY_WORKER, "identity")
    if id_data is not None:
        check(
            id_data["kl_mean"] < 0.001,
            f'identity: KL mean={id_data["kl_mean"]:.6f} < 0.001',
        )
        check(
            id_data["token_match_rate"] > 0.99,
            f'identity: token_match={id_data["token_match_rate"]:.2%} > 99%',
        )
        check(
            id_data["num_tokens"] > 0,
            f'identity: generated {id_data["num_tokens"]} tokens',
        )

    # --- Broken test (separate subprocess) ---
    br_data = _run_integration_worker(BROKEN_WORKER, "broken")
    if br_data is not None:
        check(
            br_data["kl_mean"] > 0.01,
            f'broken: KL mean={br_data["kl_mean"]:.4f} > 0.01',
        )
        check(
            br_data["token_match_rate"] < 0.99,
            f'broken: token_match={br_data["token_match_rate"]:.2%} < 99%',
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test the kb-nano bench module")
    parser.add_argument(
        "--unit-only", action="store_true",
        help="Skip the GPU integration test (run only unit tests)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  kb-nano bench module tests")
    print("=" * 60)

    test_discovery()
    test_evaluator()
    test_replacement()

    if not args.unit_only:
        test_integration()
    else:
        print(f"\n{'=' * 60}")
        print("  SKIPPED: Integration test (--unit-only)")
        print(f"{'=' * 60}")

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_pass_count} passed, {_fail_count} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
