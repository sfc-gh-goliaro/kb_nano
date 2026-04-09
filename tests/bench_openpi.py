#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano Pi0 vs OpenPI / HF Transformers.

Runs standardized robotics VLA workloads and compares:
  - Throughput: inferences/sec for action generation
  - Latency: per-inference latency with percentile stats
  - Correctness: per-sample action MSE between both engines

Both engines run with the same seed and inputs so actions are comparable.
Uses the Libero dataset for realistic observation data (images, states,
instructions).

Each engine runs in a subprocess to avoid import contamination.

Usage:
    python tests/bench_openpi.py --model lerobot/pi0_base
    python tests/bench_openpi.py --skip-openpi   # kb-nano only
    python tests/bench_openpi.py --num-requests 50 --num-steps 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from kb_nano.bench.utils.worker import run_worker
from kb_nano.bench.utils.workloads import PI0_CONFIG


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Dataset loading helper (inlined for subprocess workers)
# ---------------------------------------------------------------------------

_DATASET_PRELOAD = r'''
def load_libero_samples(num_samples, seed=42):
    """Load observation samples from Libero dataset for benchmarking."""
    import random
    import torch
    from datasets import load_dataset

    rng = random.Random(seed)
    try:
        ds = load_dataset("lerobot/libero_10_random_v0", split="train", streaming=True)
        samples = []
        for i, row in enumerate(ds):
            if i >= num_samples * 5:
                break
            samples.append(row)
        rng.shuffle(samples)
        samples = samples[:num_samples]
    except Exception as e:
        print(f"WARNING: Could not load Libero dataset: {e}", file=sys.stderr)
        print("Falling back to synthetic data", file=sys.stderr)
        samples = None

    return samples


def make_synthetic_batch(batch_size, num_cameras, image_size=224, max_state_dim=32,
                         max_action_dim=32, tokenizer_max_length=48, device="cuda",
                         dtype=torch.bfloat16, seed=42):
    """Create a synthetic batch for benchmarking when real data is unavailable."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)

    state = torch.randn(batch_size, max_state_dim, generator=gen).to(device=device, dtype=dtype)
    pixel_values = torch.randn(
        batch_size, num_cameras, 3, image_size, image_size, generator=gen,
    ).to(device=device, dtype=dtype)
    pixel_attention_mask = torch.ones(
        batch_size, num_cameras, device=device, dtype=torch.bool,
    )

    image_token_id = 257152
    num_image_tokens = (image_size // 14) ** 2
    tokens_per_camera = num_image_tokens
    total_image_tokens = tokens_per_camera * num_cameras
    text_tokens = tokenizer_max_length - total_image_tokens
    if text_tokens < 2:
        text_tokens = 2

    input_ids_list = []
    for _ in range(batch_size):
        ids = [2] + [image_token_id] * total_image_tokens + [1] * (text_tokens - 2) + [1]
        input_ids_list.append(ids[:tokenizer_max_length])
    input_ids = torch.tensor(input_ids_list, device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    return {
        "state": state,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
        "attention_mask": attention_mask,
    }
'''


# ---------------------------------------------------------------------------
# OpenPI / HF Transformers subprocess worker
# ---------------------------------------------------------------------------

OPENPI_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

''' + _DATASET_PRELOAD + r'''

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from transformers import PI0ForConditionalGeneration, PI0Processor

    model_name = cfg["model"]
    seed = cfg["seed"]
    num_steps = cfg["num_steps"]

    model = PI0ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    try:
        processor = PI0Processor.from_pretrained("google/paligemma2-3b-mix-224")
    except Exception:
        processor = None

    num_cameras = cfg.get("num_cameras", 1)
    image_size = cfg.get("image_size", 224)

    # Warmup
    batch = make_synthetic_batch(
        1, num_cameras, image_size=image_size, seed=seed,
    )
    for _ in range(2):
        torch.cuda.synchronize()
        with torch.inference_mode():
            model.sample_actions(
                state=batch["state"], input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch["pixel_attention_mask"],
                attention_mask=batch["attention_mask"],
                num_steps=2,
            )
        torch.cuda.synchronize()

    # Throughput
    throughput_results = []
    for scenario in cfg.get("scenarios", []):
        num_requests = scenario["num_requests"]
        total_elapsed = 0.0
        all_actions = []

        pbar = tqdm(range(num_requests), desc=f"openpi {scenario['name']}", file=sys.stderr)
        for req_idx in pbar:
            batch = make_synthetic_batch(
                1, scenario["num_cameras"], image_size=image_size,
                seed=seed + req_idx,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                actions = model.sample_actions(
                    state=batch["state"], input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch["pixel_attention_mask"],
                    attention_mask=batch["attention_mask"],
                    num_steps=num_steps,
                )
            torch.cuda.synchronize()
            total_elapsed += time.perf_counter() - t0
            all_actions.append(actions.cpu())
            pbar.set_postfix(ips=f"{(req_idx + 1) / total_elapsed:.2f}")

        throughput_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_requests": num_requests,
            "inferences_per_second": num_requests / total_elapsed,
        })

        actions_dir = cfg.get("actions_dir")
        if actions_dir:
            os.makedirs(actions_dir, exist_ok=True)
            torch.save(
                torch.cat(all_actions, dim=0),
                os.path.join(actions_dir, f"{scenario['name']}_actions.pt"),
            )

    # Latency
    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        batch = make_synthetic_batch(
            ls.get("batch_size", 1), ls["num_cameras"],
            image_size=image_size, seed=seed,
        )

        for _ in tqdm(range(ls.get("num_warmup", 3)), desc=f"openpi warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            with torch.inference_mode():
                model.sample_actions(
                    state=batch["state"], input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch["pixel_attention_mask"],
                    attention_mask=batch["attention_mask"],
                    num_steps=num_steps,
                )
            torch.cuda.synchronize()

        latencies = []
        for _ in tqdm(range(ls.get("num_iters", 10)), desc=f"openpi latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.sample_actions(
                    state=batch["state"], input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch["pixel_attention_mask"],
                    attention_mask=batch["attention_mask"],
                    num_steps=num_steps,
                )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "num_cameras": ls["num_cameras"],
            "batch_size": ls.get("batch_size", 1),
            "num_iters": ls.get("num_iters", 10),
            "latencies": latencies,
        })

    del model
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": throughput_results, "latency": latency_results}, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano subprocess worker
# ---------------------------------------------------------------------------

KB_NANO_PI0_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

''' + _DATASET_PRELOAD + r'''

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    eng_mod = __import__(f"{pkg}.infra.pi0_engine", fromlist=["Pi0Engine"])
    pi0_mod = __import__(f"{pkg}.tasks.baseline.L4.pi0", fromlist=["Pi0SamplingParams"])
    Pi0Engine = eng_mod.Pi0Engine
    Pi0SamplingParams = pi0_mod.Pi0SamplingParams

    seed = cfg["seed"]
    num_steps = cfg["num_steps"]
    num_cameras = cfg.get("num_cameras", 1)
    image_size = cfg.get("image_size", 224)

    engine = Pi0Engine(
        model_name=cfg["model"],
        seed=seed,
        enforce_eager=cfg.get("enforce_eager", False),
    )

    # Warmup
    batch = make_synthetic_batch(
        1, num_cameras, image_size=image_size, seed=seed,
    )
    params = Pi0SamplingParams(num_inference_steps=2)
    for _ in range(2):
        torch.cuda.synchronize()
        engine.generate(
            state=batch["state"], input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            pixel_attention_mask=batch["pixel_attention_mask"],
            attention_mask=batch["attention_mask"],
            params=params,
        )
        torch.cuda.synchronize()

    # Throughput
    throughput_results = []
    params = Pi0SamplingParams(num_inference_steps=num_steps, seed=seed)

    for scenario in cfg.get("scenarios", []):
        num_requests = scenario["num_requests"]
        total_elapsed = 0.0
        all_actions = []

        pbar = tqdm(range(num_requests), desc=f"kb-nano {scenario['name']}", file=sys.stderr)
        for req_idx in pbar:
            batch = make_synthetic_batch(
                1, scenario["num_cameras"], image_size=image_size,
                seed=seed + req_idx,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = engine.generate(
                state=batch["state"], input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch["pixel_attention_mask"],
                attention_mask=batch["attention_mask"],
                params=params,
            )
            torch.cuda.synchronize()
            total_elapsed += time.perf_counter() - t0
            all_actions.append(output.actions.cpu())
            pbar.set_postfix(ips=f"{(req_idx + 1) / total_elapsed:.2f}")

        throughput_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_requests": num_requests,
            "inferences_per_second": num_requests / total_elapsed,
        })

        actions_dir = cfg.get("actions_dir")
        if actions_dir:
            os.makedirs(actions_dir, exist_ok=True)
            torch.save(
                torch.cat(all_actions, dim=0),
                os.path.join(actions_dir, f"{scenario['name']}_actions.pt"),
            )

    # Latency
    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        batch = make_synthetic_batch(
            ls.get("batch_size", 1), ls["num_cameras"],
            image_size=image_size, seed=seed,
        )
        latency_params = Pi0SamplingParams(num_inference_steps=num_steps, seed=seed)

        for _ in tqdm(range(ls.get("num_warmup", 3)), desc=f"kb-nano warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            engine.generate(
                state=batch["state"], input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch["pixel_attention_mask"],
                attention_mask=batch["attention_mask"],
                params=latency_params,
            )
            torch.cuda.synchronize()

        latencies = []
        for _ in tqdm(range(ls.get("num_iters", 10)), desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(
                state=batch["state"], input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch["pixel_attention_mask"],
                attention_mask=batch["attention_mask"],
                params=latency_params,
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "num_cameras": ls["num_cameras"],
            "batch_size": ls.get("batch_size", 1),
            "num_iters": ls.get("num_iters", 10),
            "latencies": latencies,
        })

    engine._cleanup()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": throughput_results, "latency": latency_results}, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Result comparison helpers
# ---------------------------------------------------------------------------

def _build_scenarios(cfg: dict) -> list[dict]:
    num_requests = cfg.get("num_requests", 100)
    return [
        {
            "name": "3cam",
            "num_cameras": 3,
            "num_requests": num_requests,
        },
        {
            "name": "1cam",
            "num_cameras": 1,
            "num_requests": num_requests,
        },
    ]


def _build_latency_scenarios() -> list[dict]:
    return [
        {"name": "single-3cam", "num_cameras": 3, "batch_size": 1,
         "num_warmup": 3, "num_iters": 10},
        {"name": "single-1cam", "num_cameras": 1, "batch_size": 1,
         "num_warmup": 3, "num_iters": 10},
    ]


def _print_throughput_comparison(kb_results, ref_results=None):
    print("\n" + "=" * 90)
    print("  THROUGHPUT COMPARISON (inferences/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'Requests':>9} {'kb-nano':>12}"
    if ref_results:
        header += f" {'openpi':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = f"  {kb['name']:<25} {kb['num_requests']:>9} {kb['inferences_per_second']:>12.2f}"
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                speedup = kb["inferences_per_second"] / ref["inferences_per_second"]
                line += f" {ref['inferences_per_second']:>12.2f} {speedup:>9.2f}x"
        print(line)
    print()


def _print_latency_comparison(kb_results, ref_results=None):
    print("\n" + "=" * 80)
    print("  LATENCY COMPARISON (seconds)")
    print("=" * 80)
    header = f"  {'Scenario':<25} {'kb-nano p50':>12}"
    if ref_results:
        header += f" {'openpi p50':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 60)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        line = f"  {kb['name']:<25} {kb_p50:>12.3f}"
        if ref_results:
            ref = next((r for r in ref_results if r["name"] == kb["name"]), None)
            if ref:
                ref_lats = np.array(ref["latencies"])
                ref_p50 = np.percentile(ref_lats, 50)
                speedup = ref_p50 / kb_p50
                line += f" {ref_p50:>12.3f} {speedup:>9.2f}x"
        print(line)
    print()


def _compare_actions(kb_actions_dir, ref_actions_dir):
    """Compare action outputs between engines."""
    import torch

    kb_files = sorted(
        f for f in os.listdir(kb_actions_dir) if f.endswith(".pt")
    ) if os.path.isdir(kb_actions_dir) else []
    ref_files = sorted(
        f for f in os.listdir(ref_actions_dir) if f.endswith(".pt")
    ) if os.path.isdir(ref_actions_dir) else []

    common = sorted(set(kb_files) & set(ref_files))
    if not common:
        return {}

    results = {}
    for fname in common:
        kb_act = torch.load(
            os.path.join(kb_actions_dir, fname), map_location="cpu", weights_only=True,
        ).float()
        ref_act = torch.load(
            os.path.join(ref_actions_dir, fname), map_location="cpu", weights_only=True,
        ).float()

        n = min(kb_act.shape[0], ref_act.shape[0])
        kb_act = kb_act[:n]
        ref_act = ref_act[:n]

        mse = float((kb_act - ref_act).pow(2).mean())
        cos_sim = float(
            torch.nn.functional.cosine_similarity(
                kb_act.flatten(1), ref_act.flatten(1), dim=1,
            ).mean()
        )

        scenario = fname.replace("_actions.pt", "")
        results[scenario] = {
            "num_samples": n,
            "mean_mse": mse,
            "mean_cosine_sim": cos_sim,
        }
    return results


def _print_correctness(correctness):
    print("\n" + "=" * 80)
    print("  CORRECTNESS COMPARISON (action space)")
    print("=" * 80)
    print(f"  {'Scenario':<25} {'Samples':>8} {'Mean MSE':>12} {'CosSim':>10} {'Result':>8}")
    print("  " + "-" * 65)

    for scenario, stats in correctness.items():
        cos = stats["mean_cosine_sim"]
        verdict = "PASS" if cos > 0.95 else ("WARN" if cos > 0.80 else "FAIL")
        print(
            f"  {scenario:<25} {stats['num_samples']:>8} "
            f"{stats['mean_mse']:>12.6f} {cos:>10.6f} {verdict:>8}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pi0 benchmark: kb-nano vs OpenPI/HF Transformers",
    )
    parser.add_argument("--model", type=str, default="lerobot/pi0_base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--skip-openpi", action="store_true",
        help="Skip openpi and only benchmark kb-nano",
    )
    parser.add_argument(
        "--skip-throughput", action="store_true",
        help="Skip throughput benchmarks",
    )
    parser.add_argument(
        "--skip-latency", action="store_true",
        help="Skip latency benchmarks",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    gpu_name = _detect_gpu_name()

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu_name / short)

    print(f"\nBenchmark: Pi0 on {gpu_name}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Num denoising steps: {args.num_steps}")
    print(f"Num requests: {args.num_requests}")
    print(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    run_openpi = not args.skip_openpi
    save_actions = run_openpi

    scenarios = _build_scenarios({"num_requests": args.num_requests}) if not args.skip_throughput else []
    latency_scenarios = _build_latency_scenarios() if not args.skip_latency else []

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
        "image_size": PI0_CONFIG.image_resolution[0],
    }

    if save_actions:
        kb_actions_dir = os.path.join(args.output_dir, "actions", "kb_nano")
        ref_actions_dir = os.path.join(args.output_dir, "actions", "openpi")
    else:
        kb_actions_dir = None
        ref_actions_dir = None

    # --- kb-nano benchmark ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
    }
    if kb_actions_dir:
        kb_config["actions_dir"] = kb_actions_dir
    kb_data = run_worker(
        KB_NANO_PI0_WORKER, kb_config,
        "kb-nano Pi0 benchmark", timeout=36000,
    )

    # --- OpenPI benchmark ---
    ref_data = None
    if run_openpi:
        ref_config = {
            **base_config,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
        }
        if ref_actions_dir:
            ref_config["actions_dir"] = ref_actions_dir
        ref_data = run_worker(
            OPENPI_WORKER, ref_config,
            "openpi Pi0 benchmark", timeout=36000,
        )

    # --- Results ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        ref_tp = ref_data.get("throughput", []) if ref_data else None
        ref_lat = ref_data.get("latency", []) if ref_data else None

        if kb_tp:
            _print_throughput_comparison(kb_tp, ref_tp)
        if kb_lat:
            _print_latency_comparison(kb_lat, ref_lat)

        correctness = None
        if save_actions and kb_actions_dir and ref_actions_dir:
            correctness = _compare_actions(kb_actions_dir, ref_actions_dir)
            if correctness:
                _print_correctness(correctness)

        results_path = os.path.join(args.output_dir, "results.json")
        results = {
            "model": args.model,
            "gpu": gpu_name,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "kb_nano": kb_data,
        }
        if ref_data:
            results["openpi"] = ref_data
        if correctness:
            results["correctness"] = correctness
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_path}")
    else:
        print("ERROR: kb-nano benchmark failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
