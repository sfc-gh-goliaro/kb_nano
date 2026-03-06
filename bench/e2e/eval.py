"""Candidate-kernel evaluation: swapped kernels vs baseline.

Runs the model twice per configuration -- once with baseline kernels, once
with auto-detected candidate kernels from ``tasks/candidate/`` -- and
compares throughput and output alignment.

By default sweeps all models that have candidate kernels available.
Specify ``--model`` and/or specific ``--input-len`` / ``--output-len`` values
to narrow the evaluation.

Usage:
    # Sweep all applicable models with default workloads
    python -m kb_nano.bench.e2e eval

    # Single model, single workload
    python -m kb_nano.bench.e2e eval \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input-len 512 --output-len 128 --num-prompts 100

    # Sweep multiple workload sizes on one model
    python -m kb_nano.bench.e2e eval \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input-len 128 512 2048 --output-len 128 512
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import product

import numpy as np

from kb_nano.bench.e2e.throughput import run_kb_nano_subprocess
from kb_nano.bench.utils.datasets import SampleRequest
from kb_nano.infra.kernel_swapper import (
    BenchTarget,
    discover_candidates,
    discover_targets,
    print_candidate_summary,
)

_MODEL_KEY_TO_DEFAULT = {
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


def _resolve_models(candidates: list[tuple[BenchTarget, type]]) -> list[str]:
    """Determine which HF models to evaluate based on candidate coverage."""
    model_keys: set[str] = set()
    for target, _ in candidates:
        model_keys.update(target.models)

    all_targets = discover_targets()
    for key in sorted(model_keys):
        total_ops = [t for t in all_targets if key in t.models]
        covered_ops = [t for t, _ in candidates if key in t.models]
        missing = set(t.name for t in total_ops) - set(t.name for t in covered_ops)
        if missing:
            print(
                f"  WARNING: model {key!r} has {len(covered_ops)}/{len(total_ops)} "
                f"operators covered. Missing: {', '.join(sorted(missing))}"
            )

    models = []
    for key in sorted(model_keys):
        if key in _MODEL_KEY_TO_DEFAULT:
            models.append(_MODEL_KEY_TO_DEFAULT[key])
        else:
            print(f"  WARNING: no default HF model for key {key!r}, skipping")
    return models


def _build_requests(
    num_prompts: int, max_input_len: int, max_output_len: int, seed: int,
) -> list[SampleRequest]:
    """Build a random-token workload matching bench_throughput.py methodology."""
    rng = random.Random(seed)
    requests = []
    for _ in range(num_prompts):
        in_len = rng.randint(100, max_input_len)
        out_len = rng.randint(100, max_output_len)
        token_ids = [rng.randint(0, 10000) for _ in range(in_len)]
        requests.append(SampleRequest(
            prompt=token_ids,
            prompt_len=in_len,
            expected_output_len=out_len,
        ))
    return requests


def _check_alignment(baseline_outputs: list[dict], candidate_outputs: list[dict]) -> tuple[int, int]:
    """Compare token_ids per request. Returns (matches, total)."""
    matches = 0
    total = min(len(baseline_outputs), len(candidate_outputs))
    for b, c in zip(baseline_outputs, candidate_outputs):
        if b["token_ids"] == c["token_ids"]:
            matches += 1
    return matches, total


def _run_pair(
    requests: list[SampleRequest],
    model: str,
    tp: int,
    seed: int,
    temperature: float,
    top_p: float,
    enforce_eager: bool,
) -> dict | None:
    """Run baseline and candidate subprocesses, return combined result dict."""
    short = model.split("/")[-1]

    print(f"\n  Baseline run [{short}]...")
    baseline = run_kb_nano_subprocess(
        requests, model, tp, seed, temperature, top_p, enforce_eager,
        save_outputs=True, no_candidate_kernels=True,
    )
    if baseline is None:
        print(f"  ERROR: baseline subprocess failed for {short}")
        return None

    print(f"  Candidate run [{short}]...")
    candidate = run_kb_nano_subprocess(
        requests, model, tp, seed, temperature, top_p, enforce_eager,
        save_outputs=True, no_candidate_kernels=False,
    )
    if candidate is None:
        print(f"  ERROR: candidate subprocess failed for {short}")
        return None

    bl_tps = baseline["total_output_tokens"] / baseline["elapsed"]
    cd_tps = candidate["total_output_tokens"] / candidate["elapsed"]
    speedup = cd_tps / bl_tps if bl_tps > 0 else float("inf")

    aligned_matches, aligned_total = 0, 0
    if "outputs" in baseline and "outputs" in candidate and temperature == 0.0:
        aligned_matches, aligned_total = _check_alignment(
            baseline["outputs"], candidate["outputs"],
        )

    return {
        "baseline_tok_s": bl_tps,
        "candidate_tok_s": cd_tps,
        "speedup": speedup,
        "aligned_matches": aligned_matches,
        "aligned_total": aligned_total,
        "baseline_elapsed": baseline["elapsed"],
        "candidate_elapsed": candidate["elapsed"],
    }


def add_cli_args(parser: argparse.ArgumentParser):
    """Add eval-specific CLI arguments."""
    parser.add_argument(
        "--model", type=str, nargs="*", default=None,
        help="HuggingFace model name(s). Default: auto-detect from candidates.",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument(
        "--input-len", type=int, nargs="+", default=[1024],
        help="Max input length(s) for random workload (default: 1024)",
    )
    parser.add_argument(
        "--output-len", type=int, nargs="+", default=[1024],
        help="Max output length(s) for random workload (default: 1024)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic alignment)",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument(
        "--no-enforce-eager", dest="enforce_eager", action="store_false",
        help="Enable CUDA graphs / torch.compile",
    )
    parser.add_argument("--output-json", type=str, default=None)


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    candidates = discover_candidates()
    if not candidates:
        print("ERROR: No candidate kernels found in tasks/candidate/. Nothing to evaluate.")
        sys.exit(1)

    print_candidate_summary(candidates)

    if args.model:
        models = args.model
    else:
        models = _resolve_models(candidates)
        if not models:
            print("ERROR: No applicable models found for the discovered candidates.")
            sys.exit(1)

    input_lens = args.input_len
    output_lens = args.output_len
    combos = list(product(models, input_lens, output_lens))

    print("=" * 70)
    print("  Candidate Kernel Evaluation")
    print("=" * 70)
    print(f"  Models         : {', '.join(m.split('/')[-1] for m in models)}")
    print(f"  Input lens     : {input_lens}")
    print(f"  Output lens    : {output_lens}")
    print(f"  Configurations : {len(combos)}")
    print(f"  Prompts/config : {args.num_prompts}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  TP             : {args.tp}")
    print(f"  Seed           : {args.seed}")
    print("=" * 70)

    rows: list[dict] = []

    for model, in_len, out_len in combos:
        short = model.split("/")[-1]
        print(f"\n{'─' * 70}")
        print(f"  {short}  input={in_len}  output={out_len}")
        print(f"{'─' * 70}")

        requests = _build_requests(args.num_prompts, in_len, out_len, args.seed)
        result = _run_pair(
            requests, model, args.tp, args.seed,
            args.temperature, args.top_p, args.enforce_eager,
        )
        if result is None:
            rows.append({
                "model": short, "input_len": in_len, "output_len": out_len,
                "status": "FAILED",
            })
            continue

        rows.append({
            "model": short,
            "input_len": in_len,
            "output_len": out_len,
            "status": "OK",
            **result,
        })

    # -- Summary table --
    print(f"\n\n{'=' * 100}")
    print("  SUMMARY")
    print(f"{'=' * 100}")

    header = (
        f"  {'MODEL':<35} {'IN':>5} {'OUT':>5}  "
        f"{'BASELINE':>12}  {'CANDIDATE':>12}  {'SPEEDUP':>8}  {'ALIGNED':>8}"
    )
    print(header)
    print(f"  {'-' * 95}")

    for row in rows:
        if row["status"] == "FAILED":
            print(
                f"  {row['model']:<35} {row['input_len']:>5} {row['output_len']:>5}  "
                f"{'FAILED':>12}  {'':>12}  {'':>8}  {'':>8}"
            )
            continue

        bl_s = f"{row['baseline_tok_s']:,.0f} tok/s"
        cd_s = f"{row['candidate_tok_s']:,.0f} tok/s"
        sp_s = f"{row['speedup']:.2f}x"

        if row["aligned_total"] > 0:
            al_s = f"{row['aligned_matches']}/{row['aligned_total']}"
        elif args.temperature != 0.0:
            al_s = "n/a"
        else:
            al_s = "-"

        print(
            f"  {row['model']:<35} {row['input_len']:>5} {row['output_len']:>5}  "
            f"{bl_s:>12}  {cd_s:>12}  {sp_s:>8}  {al_s:>8}"
        )

    print(f"{'=' * 100}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n  Results saved to: {args.output_json}")
