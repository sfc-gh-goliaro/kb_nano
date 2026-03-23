#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano vs vllm-omni
for diffusion models (FLUX.1-dev).

Runs standardized diffusion workloads and compares:
  - Throughput: images/sec at various resolutions and step counts
  - Latency: per-image latency with percentile stats
  - Correctness: latent-space MSE between outputs of both engines

Prompts are drawn from the full nateraw/parti-prompts (P2) dataset (~1632
prompts), shuffled deterministically. Multiple batches per scenario provide
sustained throughput measurement analogous to the 1000-request LLM workloads.

Each engine runs in a subprocess to avoid import contamination.

Usage:
    python tests/bench_vllm_omni.py --model black-forest-labs/FLUX.1-dev
    python tests/bench_vllm_omni.py --skip-vllm-omni  # kb-nano only
    python tests/bench_vllm_omni.py --correctness-only  # compare outputs only
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

    sys.path.insert(0, cfg.get("vllm_omni_path", ""))

    from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    od_config = OmniDiffusionConfig(
        model=cfg["model"],
        dtype=torch.bfloat16,
        enforce_eager=cfg.get("enforce_eager", False),
    )
    engine = OmniDiffusion(od_config)

    warmup_params = OmniDiffusionSamplingParams(
        height=256, width=256, num_inference_steps=2,
    )
    warmup_params.seed = cfg["seed"]
    engine.generate(["warmup"], warmup_params)

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
        for batch_prompts in pbar:
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(batch_prompts, params)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)
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
import json, sys, time, torch
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
        key = (s["height"], s["width"])
        if key not in seen:
            seen.add(key)
            wp = DiffusionSamplingParams(
                height=s["height"], width=s["width"],
                num_inference_steps=2, seed=cfg["seed"],
                output_type="latent",
            )
            engine.generate(["warmup"], wp)
            torch.cuda.synchronize()

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
            output_type=scenario.get("output_type", "pil"),
        )

        total_elapsed = 0.0
        total_images = 0
        desc = f"kb-nano {scenario['name']}"
        pbar = tqdm(batches, desc=desc, unit="batch", file=sys.stderr)
        for batch_prompts in pbar:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = engine.generate(batch_prompts, params)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)
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


# ---------------------------------------------------------------------------
# Correctness workers (one per engine, returns latents for comparison)
# ---------------------------------------------------------------------------
KB_NANO_CORRECTNESS_WORKER = r'''
import json, sys, torch

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
        enforce_eager=True,
    )

    engine.warmup()

    prompts = cfg["prompts"]
    params = DiffusionSamplingParams(
        height=cfg.get("height", 512),
        width=cfg.get("width", 512),
        num_inference_steps=cfg.get("num_inference_steps", 28),
        guidance_scale=cfg.get("guidance_scale", 3.5),
        seed=cfg["seed"],
        output_type="latent",
    )

    torch.manual_seed(cfg["seed"])
    output = engine.generate(prompts, params)

    result = {}
    if output.latents is not None:
        lat = output.latents
        result["packed_latent_shape"] = list(lat.shape)
        result["latent_mean"] = float(lat.float().mean())
        result["latent_std"] = float(lat.float().std())

        pipeline = engine._get_pipeline()
        height = cfg.get("height", 512)
        width = cfg.get("width", 512)
        from kb_nano.tasks.baseline.L4.flux import FluxPipeline
        with torch.inference_mode():
            unpacked = FluxPipeline._unpack_latents(
                lat, height, width, pipeline.vae_scale_factor)
            unpacked = (unpacked / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            decoded = pipeline.vae.decode(unpacked, return_dict=False)[0]
        result["latents"] = decoded.cpu().float().tolist()
        result["latent_shape"] = list(decoded.shape)

    engine._cleanup()

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''

VLLM_OMNI_CORRECTNESS_WORKER = r'''
import json, os, sys, torch

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    sys.path.insert(0, cfg.get("vllm_omni_path", ""))

    from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    od_config = OmniDiffusionConfig(
        model=cfg["model"],
        dtype=torch.bfloat16,
        enforce_eager=True,
        output_type="latent",
    )
    engine = OmniDiffusion(od_config)

    # Warmup
    warmup_params = OmniDiffusionSamplingParams(
        height=256, width=256, num_inference_steps=2,
    )
    warmup_params.seed = cfg["seed"]
    engine.generate(["warmup"], warmup_params)

    prompts = cfg["prompts"]
    params = OmniDiffusionSamplingParams(
        height=cfg.get("height", 512),
        width=cfg.get("width", 512),
        num_inference_steps=cfg.get("num_inference_steps", 28),
        guidance_scale=cfg.get("guidance_scale", 3.5),
    )
    params.seed = cfg["seed"]

    torch.manual_seed(cfg["seed"])
    outputs = engine.generate(prompts, params)

    result = {}
    if outputs:
        import numpy as np
        latent_tensors = []
        for out in outputs:
            if hasattr(out, "images") and out.images:
                for img in out.images:
                    if isinstance(img, torch.Tensor):
                        latent_tensors.append(img.cpu().float())
        if latent_tensors:
            combined = torch.cat(latent_tensors, dim=0) if len(latent_tensors) > 1 else latent_tensors[0]
            result["latents"] = combined.numpy().tolist()
            result["latent_shape"] = list(combined.shape)
            result["latent_mean"] = float(combined.mean())
            result["latent_std"] = float(combined.std())

    del engine
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

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


def _print_correctness_results(
    kb_result: dict | None,
    vllm_result: dict | None,
    kb_result2: dict | None,
    prompts: list[str],
):
    print("\n" + "=" * 80)
    print("  CORRECTNESS RESULTS")
    print("=" * 80)

    if kb_result:
        print(f"\n  kb-nano latent shape: {kb_result.get('latent_shape')}")
        print(f"  kb-nano latent mean:  {kb_result.get('latent_mean', 'N/A'):.6f}")
        print(f"  kb-nano latent std:   {kb_result.get('latent_std', 'N/A'):.6f}")

    if vllm_result:
        print(f"\n  vllm-omni latent shape: {vllm_result.get('latent_shape')}")
        print(f"  vllm-omni latent mean:  {vllm_result.get('latent_mean', 'N/A'):.6f}")
        print(f"  vllm-omni latent std:   {vllm_result.get('latent_std', 'N/A'):.6f}")

    # Determinism check: kb-nano run1 vs run2
    if kb_result and kb_result2:
        l1 = np.array(kb_result.get("latents", []))
        l2 = np.array(kb_result2.get("latents", []))
        if l1.size > 0 and l2.size > 0 and l1.shape == l2.shape:
            mse = float(np.mean((l1 - l2) ** 2))
            print(f"\n  Determinism (kb-nano run1 vs run2):")
            print(f"    MSE: {mse:.2e}")
            if mse < 1e-6:
                print("    PASS: outputs are deterministic")
            else:
                print("    WARN: outputs differ between runs")

    # Cross-engine comparison
    if kb_result and vllm_result:
        kb_lat = np.array(kb_result.get("latents", []))
        vllm_lat = np.array(vllm_result.get("latents", []))
        if kb_lat.size > 0 and vllm_lat.size > 0:
            # Shapes may differ due to packing; compare flattened
            kb_flat = kb_lat.flatten()
            vllm_flat = vllm_lat.flatten()
            min_len = min(len(kb_flat), len(vllm_flat))
            if min_len > 0:
                kb_flat = kb_flat[:min_len]
                vllm_flat = vllm_flat[:min_len]
                mse = float(np.mean((kb_flat - vllm_flat) ** 2))
                cos_sim = float(
                    np.dot(kb_flat, vllm_flat)
                    / (np.linalg.norm(kb_flat) * np.linalg.norm(vllm_flat) + 1e-12)
                )
                print(f"\n  Cross-engine (kb-nano vs vllm-omni):")
                print(f"    Latent MSE:        {mse:.2e}")
                print(f"    Cosine similarity: {cos_sim:.6f}")
                if cos_sim > 0.99:
                    print("    PASS: outputs match closely")
                elif cos_sim > 0.95:
                    print("    WARN: outputs similar but not identical")
                else:
                    print("    FAIL: outputs diverge significantly")
        else:
            print("\n  Could not compare cross-engine latents (empty data)")

    print(f"\n  Prompts used ({len(prompts)} from parti-prompts P2):")
    for i, p in enumerate(prompts):
        print(f"    [{i+1}] {p[:80]}{'...' if len(p) > 80 else ''}")
    print()


def main():
    parser = argparse.ArgumentParser(description="FLUX benchmark: kb-nano vs vllm-omni")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-vllm-omni", action="store_true",
                        help="Skip vllm-omni and only benchmark kb-nano")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Only run correctness check (smaller workload)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for all scenarios")
    parser.add_argument("--vllm-omni-path", type=str, default=None,
                        help="Path to vllm-omni repo root")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for worker log files (use tail -f to monitor)")
    args = parser.parse_args()

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        main_log = os.path.join(args.log_dir, "bench_main.log")
        log_fh = open(main_log, "w")
        sys.stdout = log_fh
        sys.stderr = log_fh
        print(f"Main log: {main_log}", file=sys.__stderr__, flush=True)

    gpu_name = _detect_gpu_name()
    print(f"\nBenchmark: FLUX.1-dev on {gpu_name}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Enforce eager: {args.enforce_eager}")

    # Load prompts from parti-prompts dataset
    bench_prompts = _get_bench_prompts(args.seed)
    print(f"Loaded {len(bench_prompts)} prompts from parti-prompts (P2)")

    vllm_omni_path = args.vllm_omni_path
    if vllm_omni_path is None:
        candidates = [
            str(_PROJECT_ROOT / "vllm_repo" / "vllm-omni"),
            str(_PROJECT_ROOT / "vllm-omni"),
            os.path.expanduser("~/vllm-omni"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                vllm_omni_path = c
                break

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
    }

    # --- Correctness check ---
    if args.correctness_only:
        print("\n--- Correctness Check ---")
        correctness_config = {
            **base_config,
            "prompts": bench_prompts[:4],
            "height": 512,
            "width": 512,
            "num_inference_steps": 28,
        }

        # Run kb-nano (twice for determinism check)
        kb_result = run_worker(
            KB_NANO_CORRECTNESS_WORKER, correctness_config,
            "kb-nano correctness", timeout=600, log_dir=args.log_dir,
        )
        kb_result2 = run_worker(
            KB_NANO_CORRECTNESS_WORKER, correctness_config,
            "kb-nano correctness (run 2)", timeout=600, log_dir=args.log_dir,
        )

        # Run vllm-omni for cross-engine comparison
        vllm_result = None
        if not args.skip_vllm_omni and vllm_omni_path:
            vllm_correctness_config = {
                **correctness_config,
                "vllm_omni_path": vllm_omni_path,
            }
            vllm_result = run_worker(
                VLLM_OMNI_CORRECTNESS_WORKER, vllm_correctness_config,
                "vllm-omni correctness", timeout=600, log_dir=args.log_dir,
            )

        _print_correctness_results(
            kb_result, vllm_result, kb_result2, bench_prompts[:4],
        )
        return

    # Build scenarios
    scenarios = _build_throughput_scenarios(bench_prompts, args.batch_size)
    latency_scenarios = _build_latency_scenarios(bench_prompts)

    # --- kb-nano benchmark ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
    }
    kb_data = run_worker(
        KB_NANO_DIFFUSION_WORKER, kb_config,
        "kb-nano diffusion benchmark", timeout=36000, log_dir=args.log_dir,
    )

    # --- vllm-omni benchmark ---
    vllm_data = None
    if not args.skip_vllm_omni and vllm_omni_path:
        vllm_config = {
            **base_config,
            "vllm_omni_path": vllm_omni_path,
            "scenarios": scenarios,
            "latency_scenarios": latency_scenarios,
        }
        vllm_data = run_worker(
            VLLM_OMNI_WORKER, vllm_config,
            "vllm-omni diffusion benchmark", timeout=36000, log_dir=args.log_dir,
        )
    elif not args.skip_vllm_omni:
        print("\n  WARNING: vllm-omni not found, skipping comparison.")
        print("  Set --vllm-omni-path or ensure vllm-omni is at a standard location.\n")

    # --- Print results ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        vllm_tp = vllm_data.get("throughput", []) if vllm_data else None
        vllm_lat = vllm_data.get("latency", []) if vllm_data else None

        _print_throughput_comparison(kb_tp, vllm_tp)
        _print_latency_comparison(kb_lat, vllm_lat)

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
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    else:
        print("ERROR: kb-nano benchmark failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
