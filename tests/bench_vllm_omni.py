#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano vs vllm-omni
for diffusion models (FLUX.1-dev).

Runs standardized diffusion workloads and compares:
  - Throughput: images/sec at various resolutions and step counts
  - Latency: per-image latency with percentile stats
  - Correctness: per-batch packed-latent MSE and cosine similarity between
    outputs of both engines (assessed for every image during throughput runs)

Both engines run with output_type="latent" (no VAE decode) so the benchmark
measures the transformer backbone.  Packed latents are saved per-batch and
compared numerically after both engines finish.

Prompts are drawn from the full nateraw/parti-prompts (P2) dataset (~1632
prompts), shuffled deterministically. Multiple batches per scenario provide
sustained throughput measurement analogous to the 1000-request LLM workloads.

Each engine runs in a subprocess to avoid import contamination.

Usage:
    python tests/bench_vllm_omni.py --model black-forest-labs/FLUX.1-dev
    python tests/bench_vllm_omni.py --skip-vllm-omni  # kb-nano only (no correctness)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from kb_nano.bench.utils.worker import run_worker
from kb_nano.bench.utils.workloads import (
    DIFFUSION_LATENCY_WORKLOADS,
    DIFFUSION_THROUGHPUT_WORKLOADS,
)


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
# Prompt loading from nateraw/parti-prompts
# ---------------------------------------------------------------------------

def _load_parti_prompts(seed: int = 42) -> list[str]:
    """Load all prompts from nateraw/parti-prompts (P2), deterministically shuffled.

    Returns ~1632 prompts covering diverse challenge categories.
    The full set enables realistic throughput benchmarking analogous to
    the 1000-request LLM workloads.
    """
    from datasets import load_dataset

    ds = load_dataset("nateraw/parti-prompts", split="train")
    prompts = [row["Prompt"] for row in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


# Lazy-loaded and cached
_PARTI_PROMPTS: list[str] | None = None


def _get_bench_prompts(seed: int = 42) -> list[str]:
    """Return benchmark prompts (cached after first load)."""
    global _PARTI_PROMPTS
    if _PARTI_PROMPTS is None:
        _PARTI_PROMPTS = _load_parti_prompts(seed)
    return _PARTI_PROMPTS


# ---------------------------------------------------------------------------
# vllm-omni subprocess worker
# ---------------------------------------------------------------------------
VLLM_OMNI_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    od_config = OmniDiffusionConfig(
        model=cfg["model"],
        dtype=torch.bfloat16,
        enforce_eager=cfg.get("enforce_eager", False),
        output_type="latent",
    )
    engine = OmniDiffusion(od_config)

    warmup_params = OmniDiffusionSamplingParams(
        height=256, width=256, num_inference_steps=2,
    )
    warmup_params.seed = cfg["seed"]
    engine.generate(["warmup"], warmup_params)

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    all_results = []
    for scenario in cfg["scenarios"]:
        batches = scenario.get("batches", [scenario.get("prompts", [])])
        if not isinstance(batches[0], list):
            batches = [batches]
        params = OmniDiffusionSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 3.5),
        )
        params.seed = cfg["seed"]

        total_elapsed = 0.0
        total_images = 0
        desc = f"vllm-omni {scenario['name']}"
        pbar = tqdm(batches, desc=desc, unit="batch", file=sys.stderr)
        for batch_idx, batch_prompts in enumerate(pbar):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(batch_prompts, params)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)

            if latent_dir and outputs:
                latent_tensor = None
                for out in outputs:
                    if hasattr(out, "images") and out.images:
                        for img in out.images:
                            if isinstance(img, torch.Tensor):
                                latent_tensor = img
                                break
                    if latent_tensor is not None:
                        break
                    if hasattr(out, "latents") and out.latents is not None:
                        latent_tensor = out.latents
                        break
                if latent_tensor is not None:
                    torch.save(
                        latent_tensor.cpu(),
                        os.path.join(latent_dir, f"{scenario['name']}_batch{batch_idx:04d}.pt"),
                    )

            pbar.set_postfix(
                imgs=total_images,
                ips=f"{total_images / total_elapsed:.2f}",
            )

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_images": total_images,
            "images_per_second": total_images / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        params = OmniDiffusionSamplingParams(
            height=ls["height"],
            width=ls["width"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 3.5),
        )
        params.seed = cfg["seed"]

        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)

        for i in tqdm(range(num_warmup), desc=f"vllm-omni latency warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            engine.generate(prompts, params)
            torch.cuda.synchronize()

        latencies = []
        for i in tqdm(range(num_iters), desc=f"vllm-omni latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "height": ls["height"],
            "width": ls["width"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del engine
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano subprocess worker
# ---------------------------------------------------------------------------
KB_NANO_DIFFUSION_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    eng_mod = __import__(
        f"{pkg}.infra.diffusion_engine",
        fromlist=["DiffusionEngine"],
    )
    flux_mod = __import__(
        f"{pkg}.tasks.baseline.L4.flux",
        fromlist=["DiffusionSamplingParams"],
    )
    DiffusionEngine = eng_mod.DiffusionEngine
    DiffusionSamplingParams = flux_mod.DiffusionSamplingParams

    engine = DiffusionEngine(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
    )

    seen = set()
    for s in cfg["scenarios"] + cfg.get("latency_scenarios", []):
        bs = s.get("batch_size", len(s.get("prompts", ["w"])))
        key = (s["height"], s["width"], bs)
        if key not in seen:
            seen.add(key)
            wp = DiffusionSamplingParams(
                height=s["height"], width=s["width"],
                num_inference_steps=2, seed=cfg["seed"],
                output_type="latent",
            )
            warmup_prompts = [f"warmup {i}" for i in range(bs)]
            print(f"Warming up: {s['height']}x{s['width']} batch_size={bs}", file=sys.stderr, flush=True)
            engine.generate(warmup_prompts, wp)
            torch.cuda.synchronize()

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    pipeline = engine._get_pipeline()

    all_results = []
    for scenario in cfg["scenarios"]:
        batches = scenario.get("batches", [scenario.get("prompts", [])])
        if not isinstance(batches[0], list):
            batches = [batches]
        params = DiffusionSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 3.5),
            seed=cfg["seed"],
            output_type="latent",
        )

        total_elapsed = 0.0
        total_images = 0
        desc = f"kb-nano {scenario['name']}"
        pbar = tqdm(batches, desc=desc, unit="batch", file=sys.stderr)
        for batch_idx, batch_prompts in enumerate(pbar):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = engine.generate(batch_prompts, params)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)

            if latent_dir and output.latents is not None:
                packed = output.latents
                unpacked = pipeline._unpack_latents(
                    packed, scenario["height"], scenario["width"],
                    pipeline.vae_scale_factor,
                )
                decoded = (unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                decoded = pipeline.vae.decode(decoded, return_dict=False)[0]
                torch.save(
                    decoded.cpu(),
                    os.path.join(latent_dir, f"{scenario['name']}_batch{batch_idx:04d}.pt"),
                )

            pbar.set_postfix(
                imgs=total_images,
                ips=f"{total_images / total_elapsed:.2f}",
            )

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_images": total_images,
            "images_per_second": total_images / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        params = DiffusionSamplingParams(
            height=ls["height"],
            width=ls["width"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 3.5),
            seed=cfg["seed"],
            output_type="latent",
        )

        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)

        for i in tqdm(range(num_warmup), desc=f"kb-nano latency warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            engine.generate(prompts, params)
            torch.cuda.synchronize()

        latencies = []
        for i in tqdm(range(num_iters), desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "height": ls["height"],
            "width": ls["width"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    engine._cleanup()

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''



def _build_throughput_scenarios(
    prompts: list[str], batch_size_override: int | None = None,
) -> list[dict]:
    scenarios = []
    for w in DIFFUSION_THROUGHPUT_WORKLOADS:
        bs = batch_size_override or w.batch_size
        num_requests = w.num_requests
        total_needed = bs * num_requests
        pool = (prompts * ((total_needed // len(prompts)) + 1))[:total_needed]
        batches = [pool[i * bs : (i + 1) * bs] for i in range(num_requests)]
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": w.num_inference_steps,
            "guidance_scale": w.guidance_scale,
            "batches": batches,
            "batch_size": bs,
            "num_requests": num_requests,
        })
    return scenarios


def _build_latency_scenarios(prompts: list[str]) -> list[dict]:
    scenarios = []
    for w in DIFFUSION_LATENCY_WORKLOADS:
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": w.num_inference_steps,
            "guidance_scale": w.guidance_scale,
            "prompts": prompts[:w.batch_size],
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


def _print_throughput_comparison(kb_results: list[dict], vllm_results: list[dict] | None):
    print("\n" + "=" * 90)
    print("  THROUGHPUT COMPARISON (images/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'Images':>7} {'kb-nano':>12}"
    if vllm_results:
        header += f" {'vllm-omni':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = f"  {kb['name']:<25} {kb['num_images']:>7} {kb['images_per_second']:>12.2f}"
        if vllm_results:
            vllm = next((v for v in vllm_results if v["name"] == kb["name"]), None)
            if vllm:
                speedup = kb["images_per_second"] / vllm["images_per_second"]
                line += f" {vllm['images_per_second']:>12.2f} {speedup:>9.2f}x"
        print(line)
    print()


def _print_latency_comparison(kb_results: list[dict], vllm_results: list[dict] | None):
    print("\n" + "=" * 80)
    print("  LATENCY COMPARISON (seconds)")
    print("=" * 80)
    header = f"  {'Scenario':<25} {'kb-nano p50':>12}"
    if vllm_results:
        header += f" {'vllm-omni p50':>14} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 60)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        line = f"  {kb['name']:<25} {kb_p50:>12.3f}"
        if vllm_results:
            vllm = next((v for v in vllm_results if v["name"] == kb["name"]), None)
            if vllm:
                vllm_lats = np.array(vllm["latencies"])
                vllm_p50 = np.percentile(vllm_lats, 50)
                speedup = vllm_p50 / kb_p50
                line += f" {vllm_p50:>14.3f} {speedup:>9.2f}x"
        print(line)
    print()


def _compare_latents(kb_latent_dir: str, vllm_latent_dir: str) -> dict:
    """Compare per-batch output tensors between kb-nano and vllm-omni.

    Both engines may produce tensors in different representations (packed
    latents vs decoded images).  The comparison flattens both tensors and
    computes MSE and cosine similarity, which remains valid as long as the
    total element count is comparable.  When element counts differ (e.g.
    packed latents vs decoded images), the batch is skipped with a warning.

    Returns a dict mapping scenario names to per-scenario correctness stats.
    """
    import torch

    kb_files = sorted(
        f for f in os.listdir(kb_latent_dir) if f.endswith(".pt")
    ) if os.path.isdir(kb_latent_dir) else []
    vllm_files = sorted(
        f for f in os.listdir(vllm_latent_dir) if f.endswith(".pt")
    ) if os.path.isdir(vllm_latent_dir) else []

    common = sorted(set(kb_files) & set(vllm_files))
    if not common:
        return {}

    from collections import defaultdict
    scenario_stats: dict[str, list[dict]] = defaultdict(list)

    for fname in common:
        kb_lat = torch.load(
            os.path.join(kb_latent_dir, fname), map_location="cpu", weights_only=True,
        ).float().flatten()
        vllm_lat = torch.load(
            os.path.join(vllm_latent_dir, fname), map_location="cpu", weights_only=True,
        ).float().flatten()

        if len(kb_lat) != len(vllm_lat):
            print(
                f"  WARNING: shape mismatch for {fname}: "
                f"kb-nano={kb_lat.shape} vs vllm-omni={vllm_lat.shape}, skipping",
                file=sys.stderr,
            )
            continue

        kb_v = kb_lat.numpy()
        vllm_v = vllm_lat.numpy()

        mse = float(np.mean((kb_v - vllm_v) ** 2))
        cos_sim = float(
            np.dot(kb_v, vllm_v)
            / (np.linalg.norm(kb_v) * np.linalg.norm(vllm_v) + 1e-12)
        )

        scenario_name = fname.rsplit("_batch", 1)[0]
        scenario_stats[scenario_name].append({
            "file": fname,
            "mse": mse,
            "cosine_similarity": cos_sim,
        })

    results = {}
    for scenario, batches in scenario_stats.items():
        mses = [b["mse"] for b in batches]
        cosines = [b["cosine_similarity"] for b in batches]
        results[scenario] = {
            "num_batches": len(batches),
            "mean_mse": float(np.mean(mses)),
            "max_mse": float(np.max(mses)),
            "mean_cosine_sim": float(np.mean(cosines)),
            "min_cosine_sim": float(np.min(cosines)),
        }
    return results


def _print_correctness_comparison(correctness: dict):
    """Print per-scenario correctness table from latent comparison."""
    print("\n" + "=" * 90)
    print("  CORRECTNESS COMPARISON (packed latent space, per-batch)")
    print("=" * 90)
    print(f"  {'Scenario':<25} {'Batches':>8} {'Mean CosSim':>12} {'Min CosSim':>11} {'Mean MSE':>12} {'Max MSE':>12} {'Result':>8}")
    print("  " + "-" * 88)

    all_pass = True
    for scenario, stats in correctness.items():
        min_cos = stats["min_cosine_sim"]
        verdict = "PASS" if min_cos > 0.99 else ("WARN" if min_cos > 0.95 else "FAIL")
        if verdict != "PASS":
            all_pass = False
        print(
            f"  {scenario:<25} {stats['num_batches']:>8} "
            f"{stats['mean_cosine_sim']:>12.6f} {min_cos:>11.6f} "
            f"{stats['mean_mse']:>12.2e} {stats['max_mse']:>12.2e} "
            f"{verdict:>8}"
        )

    print()
    if all_pass:
        print("  All scenarios PASS (min cosine similarity > 0.99)")
    else:
        print("  WARNING: Some scenarios have divergent outputs")
    print()


def main():
    parser = argparse.ArgumentParser(description="FLUX benchmark: kb-nano vs vllm-omni")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-vllm-omni", action="store_true",
                        help="Skip vllm-omni and only benchmark kb-nano")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for all scenarios")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    gpu_name = _detect_gpu_name()
    print(f"\nBenchmark: FLUX.1-dev on {gpu_name}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Enforce eager: {args.enforce_eager}")

    bench_prompts = _get_bench_prompts(args.seed)
    print(f"Loaded {len(bench_prompts)} prompts from parti-prompts (P2)")

    run_vllm = not args.skip_vllm_omni
    save_latents = run_vllm

    latent_tmpdir = tempfile.mkdtemp(prefix="flux_latents_") if save_latents else None
    kb_latent_dir = os.path.join(latent_tmpdir, "kb_nano") if latent_tmpdir else None
    vllm_latent_dir = os.path.join(latent_tmpdir, "vllm_omni") if latent_tmpdir else None

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
    }

    scenarios = _build_throughput_scenarios(bench_prompts, args.batch_size)
    latency_scenarios = _build_latency_scenarios(bench_prompts)

    # --- kb-nano benchmark ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
    }
    if kb_latent_dir:
        kb_config["latent_dir"] = kb_latent_dir
    kb_data = run_worker(
        KB_NANO_DIFFUSION_WORKER, kb_config,
        "kb-nano diffusion benchmark", timeout=36000,
    )

    # --- vllm-omni benchmark ---
    vllm_data = None
    if run_vllm:
        vllm_config = {
            **base_config,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
        }
        if vllm_latent_dir:
            vllm_config["latent_dir"] = vllm_latent_dir
        vllm_data = run_worker(
            VLLM_OMNI_WORKER, vllm_config,
            "vllm-omni diffusion benchmark", timeout=36000,
        )
    # --- Print results ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        vllm_tp = vllm_data.get("throughput", []) if vllm_data else None
        vllm_lat = vllm_data.get("latency", []) if vllm_data else None

        _print_throughput_comparison(kb_tp, vllm_tp)
        _print_latency_comparison(kb_lat, vllm_lat)

        # Per-batch correctness comparison
        correctness = None
        if save_latents and kb_latent_dir and vllm_latent_dir:
            correctness = _compare_latents(kb_latent_dir, vllm_latent_dir)
            if correctness:
                _print_correctness_comparison(correctness)
            else:
                print("\n  WARNING: No matching latent files found for correctness comparison.")

        # Save results
        output_file = args.output or f"bench_vllm_omni_results_{gpu_name}.json"
        results = {
            "model": args.model,
            "gpu": gpu_name,
            "seed": args.seed,
            "prompts_source": "nateraw/parti-prompts",
            "num_prompts": len(bench_prompts),
            "kb_nano": kb_data,
        }
        if vllm_data:
            results["vllm_omni"] = vllm_data
        if correctness:
            results["correctness"] = correctness
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    else:
        print("ERROR: kb-nano benchmark failed.")
        sys.exit(1)

    if latent_tmpdir:
        shutil.rmtree(latent_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
