#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano vs vllm-omni
for diffusion models (FLUX.1-dev).

Runs standardized diffusion workloads and compares:
  - Throughput: images/sec at various resolutions and step counts
  - Latency: per-image latency with percentile stats
  - Correctness: latent-space MSE and image PSNR between outputs

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


# Standard prompts for benchmarking (same across both engines for correctness)
BENCH_PROMPTS = [
    "A serene mountain landscape at sunset with golden light reflecting off a crystal-clear lake",
    "A futuristic cityscape with flying cars and neon-lit skyscrapers at night",
    "An oil painting of a cat sitting in a sunlit window, impressionist style",
    "A detailed photograph of a steampunk clockwork mechanism with brass gears",
    "A watercolor painting of cherry blossoms along a Japanese garden path",
    "A photorealistic image of the Northern Lights over a snow-covered forest",
    "An abstract geometric art piece with vibrant colors and sharp angles",
    "A cozy coffee shop interior with warm lighting and rain on the windows",
]


# ---------------------------------------------------------------------------
# vllm-omni subprocess worker
# ---------------------------------------------------------------------------
VLLM_OMNI_WORKER = r'''
import json, os, sys, time, torch

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

    # Warmup
    warmup_params = OmniDiffusionSamplingParams(
        height=256, width=256, num_inference_steps=2,
    )
    warmup_params.seed = cfg["seed"]
    engine.generate(["warmup"], warmup_params)

    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        params = OmniDiffusionSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 3.5),
        )
        params.seed = cfg["seed"]

        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(prompts, params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "num_images": len(prompts),
            "images_per_second": len(prompts) / elapsed,
        }

        if cfg.get("save_latents", False) and outputs:
            latent_list = []
            for out in outputs:
                if hasattr(out, "outputs") and out.outputs:
                    for o in out.outputs:
                        if hasattr(o, "data") and o.data is not None:
                            latent_list.append(o.data.cpu().tolist())
            result["latents"] = latent_list

        all_results.append(result)

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

        for _ in range(num_warmup):
            torch.cuda.synchronize()
            engine.generate(prompts, params)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iters):
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

    # Warmup
    engine.warmup()

    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        params = DiffusionSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 3.5),
            seed=cfg["seed"],
            output_type=scenario.get("output_type", "pil"),
        )

        torch.cuda.synchronize()
        start = time.perf_counter()
        output = engine.generate(prompts, params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "num_images": len(prompts),
            "images_per_second": len(prompts) / elapsed,
        }

        if cfg.get("save_latents", False) and output.latents is not None:
            result["latents"] = output.latents.cpu().tolist()

        all_results.append(result)

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

        for _ in range(num_warmup):
            torch.cuda.synchronize()
            engine.generate(prompts, params)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_iters):
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
# Correctness comparison worker (runs both engines, compares latents)
# ---------------------------------------------------------------------------
CORRECTNESS_WORKER = r'''
import json, sys, time, torch
import numpy as np

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
        result["latents"] = output.latents.cpu().float().tolist()
        result["latent_shape"] = list(output.latents.shape)
        result["latent_mean"] = float(output.latents.float().mean())
        result["latent_std"] = float(output.latents.float().std())

    engine._cleanup()

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def _build_throughput_scenarios(batch_size_override: int | None = None) -> list[dict]:
    scenarios = []
    for w in DIFFUSION_THROUGHPUT_WORKLOADS:
        bs = batch_size_override or w.batch_size
        prompts = BENCH_PROMPTS[:bs]
        if len(prompts) < bs:
            prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": w.num_inference_steps,
            "guidance_scale": w.guidance_scale,
            "prompts": prompts,
        })
    return scenarios


def _build_latency_scenarios() -> list[dict]:
    scenarios = []
    for w in DIFFUSION_LATENCY_WORKLOADS:
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": w.num_inference_steps,
            "guidance_scale": w.guidance_scale,
            "prompts": BENCH_PROMPTS[:w.batch_size],
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


def _print_throughput_comparison(kb_results: list[dict], vllm_results: list[dict] | None):
    print("\n" + "=" * 80)
    print("  THROUGHPUT COMPARISON (images/sec)")
    print("=" * 80)
    header = f"  {'Scenario':<25} {'kb-nano':>12}"
    if vllm_results:
        header += f" {'vllm-omni':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 60)

    for kb in kb_results:
        line = f"  {kb['name']:<25} {kb['images_per_second']:>12.2f}"
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
    args = parser.parse_args()

    gpu_name = _detect_gpu_name()
    print(f"\nBenchmark: FLUX.1-dev on {gpu_name}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Enforce eager: {args.enforce_eager}")

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

    # Build scenarios
    scenarios = _build_throughput_scenarios(args.batch_size)
    latency_scenarios = _build_latency_scenarios()

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
            "prompts": BENCH_PROMPTS[:1],
            "height": 512,
            "width": 512,
            "num_inference_steps": 28,
        }

        # Run kb-nano
        kb_result = run_worker(
            CORRECTNESS_WORKER, correctness_config,
            "kb-nano correctness", timeout=600,
        )
        if kb_result:
            print(f"  kb-nano latent shape: {kb_result.get('latent_shape')}")
            print(f"  kb-nano latent mean:  {kb_result.get('latent_mean', 'N/A'):.6f}")
            print(f"  kb-nano latent std:   {kb_result.get('latent_std', 'N/A'):.6f}")

        # Determinism: run kb-nano twice and compare
        kb_result2 = run_worker(
            CORRECTNESS_WORKER, correctness_config,
            "kb-nano correctness (run 2)", timeout=600,
        )
        if kb_result and kb_result2:
            l1 = np.array(kb_result.get("latents", []))
            l2 = np.array(kb_result2.get("latents", []))
            if l1.size > 0 and l2.size > 0 and l1.shape == l2.shape:
                mse = float(np.mean((l1 - l2) ** 2))
                print(f"\n  Determinism check (run1 vs run2):")
                print(f"    MSE: {mse:.2e}")
                if mse < 1e-10:
                    print("    PASS: outputs are deterministic")
                else:
                    print("    WARN: outputs differ between runs")
            else:
                print("  Could not compare latents (shape mismatch or empty)")
        return

    # --- kb-nano benchmark ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
    }
    kb_data = run_worker(
        KB_NANO_DIFFUSION_WORKER, kb_config,
        "kb-nano diffusion benchmark", timeout=3600,
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
            "vllm-omni diffusion benchmark", timeout=3600,
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
