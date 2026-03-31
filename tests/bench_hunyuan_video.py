#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano vs vllm-omni
for HunyuanVideo-1.5 text-to-video diffusion.

Runs standardized video diffusion workloads and compares:
  - Throughput: videos/sec at 480p with varying frame counts
  - Latency: per-video latency with percentile stats
  - Correctness: per-prompt latent MSE and cosine similarity between
    outputs of both engines (assessed for every prompt during throughput)

Both engines run with output_type="latent" (no VAE decode) so the benchmark
measures the transformer backbone.  Latents are saved per-prompt and
compared numerically after both engines finish.

Prompts are drawn from Meta's Movie Gen Video Bench dataset (~1003 prompts
covering human activity, animals, nature, physics, and unusual subjects),
shuffled deterministically.

Each engine runs in a subprocess to avoid import contamination.

Usage:
    python tests/bench_hunyuan_video.py
    python tests/bench_hunyuan_video.py --skip-vllm-omni   # kb-nano only
    python tests/bench_hunyuan_video.py --skip-latency      # throughput only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from kb_nano.bench.utils.worker import run_worker
from kb_nano.bench.utils.workloads import (
    HUNYUAN_VIDEO_CONFIG,
    VIDEO_DIFFUSION_LATENCY_WORKLOADS,
    VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS,
)

MODEL = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"


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
# Prompt loading from Movie Gen Video Bench
# ---------------------------------------------------------------------------

_MOVIE_GEN_PROMPTS: list[str] | None = None


def _load_movie_gen_prompts(seed: int = 42) -> list[str]:
    """Load all ~1003 prompts from Movie Gen Video Bench, deterministically shuffled."""
    from datasets import load_dataset

    ds = load_dataset(
        "meta-ai-for-media-research/movie_gen_video_bench_no_generations",
        split="test",
    )
    prompts = [row["prompt"] for row in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def _get_bench_prompts(seed: int = 42) -> list[str]:
    global _MOVIE_GEN_PROMPTS
    if _MOVIE_GEN_PROMPTS is None:
        _MOVIE_GEN_PROMPTS = _load_movie_gen_prompts(seed)
    return _MOVIE_GEN_PROMPTS


# ---------------------------------------------------------------------------
# kb-nano subprocess worker
# ---------------------------------------------------------------------------
KB_NANO_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])

    from kb_nano.infra.diffusion_engine import DiffusionEngine
    from kb_nano.tasks.baseline.L4.hunyuan_video import (
        HunyuanVideoDiffusionSamplingParams,
    )

    engine = DiffusionEngine(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=True,
    )

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    # Warmup at each unique resolution/frame-count
    seen = set()
    for s in cfg["scenarios"] + cfg.get("latency_scenarios", []):
        key = (s["height"], s["width"], s["num_frames"])
        if key not in seen:
            seen.add(key)
            wp = HunyuanVideoDiffusionSamplingParams(
                height=s["height"], width=s["width"],
                num_frames=s["num_frames"],
                num_inference_steps=2,
                guidance_scale=s.get("guidance_scale", 6.0),
                seed=cfg["seed"], output_type="latent",
            )
            print(f"Warming up: {s['height']}x{s['width']} {s['num_frames']}f",
                  file=sys.stderr, flush=True)
            engine.generate(["warmup"], wp)
            torch.cuda.synchronize()

    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        params = HunyuanVideoDiffusionSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_frames=scenario["num_frames"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 6.0),
            seed=cfg["seed"],
            output_type="latent",
        )

        total_elapsed = 0.0
        total_videos = 0
        desc = f"kb-nano {scenario['name']}"
        pbar = tqdm(enumerate(prompts), total=len(prompts),
                    desc=desc, unit="vid", file=sys.stderr)
        for pi, prompt in pbar:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = engine.generate([prompt], params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_elapsed += elapsed
            total_videos += 1

            if latent_dir and output.latents is not None:
                torch.save(
                    output.latents.cpu(),
                    os.path.join(
                        latent_dir,
                        f"{scenario['name']}_prompt{pi:04d}.pt",
                    ),
                )

            pbar.set_postfix(
                vids=total_videos,
                vps=f"{total_videos / total_elapsed:.3f}",
            )

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_videos": total_videos,
            "videos_per_second": total_videos / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompt = ls["prompt"]
        params = HunyuanVideoDiffusionSamplingParams(
            height=ls["height"],
            width=ls["width"],
            num_frames=ls["num_frames"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 6.0),
            seed=cfg["seed"],
            output_type="latent",
        )

        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)

        for i in tqdm(range(num_warmup),
                      desc=f"kb-nano warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            engine.generate([prompt], params)
            torch.cuda.synchronize()

        latencies = []
        for i in tqdm(range(num_iters),
                      desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate([prompt], params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "height": ls["height"],
            "width": ls["width"],
            "num_frames": ls["num_frames"],
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
# vllm-omni subprocess worker
# ---------------------------------------------------------------------------
VLLM_OMNI_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

def _extract_latent(output):
    """Extract latent tensor from OmniRequestOutput (list or single)."""
    items = output if isinstance(output, list) else [output]
    for item in items:
        if hasattr(item, "latents") and item.latents is not None:
            return item.latents
        if hasattr(item, "images") and item.images:
            for img in item.images:
                if isinstance(img, torch.Tensor):
                    return img
    return None

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    model = cfg["model"]
    seed = cfg["seed"]

    print("[vllm-omni] Creating Omni engine...", file=sys.stderr, flush=True)
    omni = Omni(model=model, enforce_eager=True)
    print("[vllm-omni] Engine ready", file=sys.stderr, flush=True)

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    def _make_params(scenario_or_ls):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        p = OmniDiffusionSamplingParams(
            height=scenario_or_ls["height"],
            width=scenario_or_ls["width"],
            num_frames=scenario_or_ls["num_frames"],
            num_inference_steps=scenario_or_ls["num_inference_steps"],
            guidance_scale=scenario_or_ls.get("guidance_scale", 6.0),
            generator=generator,
            output_type="latent",
        )
        return p

    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        total_elapsed = 0.0
        total_videos = 0
        desc = f"vllm-omni {scenario['name']}"
        pbar = tqdm(enumerate(prompts), total=len(prompts),
                    desc=desc, unit="vid", file=sys.stderr)
        for pi, prompt in pbar:
            params = _make_params(scenario)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = omni.generate({"prompt": prompt}, params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_elapsed += elapsed
            total_videos += 1

            if latent_dir:
                latent_tensor = _extract_latent(output)
                if latent_tensor is not None:
                    torch.save(
                        latent_tensor.cpu(),
                        os.path.join(
                            latent_dir,
                            f"{scenario['name']}_prompt{pi:04d}.pt",
                        ),
                    )

            pbar.set_postfix(
                vids=total_videos,
                vps=f"{total_videos / total_elapsed:.3f}",
            )

        all_results.append({
            "name": scenario["name"],
            "elapsed": total_elapsed,
            "num_videos": total_videos,
            "videos_per_second": total_videos / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompt = ls["prompt"]
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)

        for i in tqdm(range(num_warmup),
                      desc=f"vllm-omni warmup {ls['name']}", file=sys.stderr):
            params = _make_params(ls)
            torch.cuda.synchronize()
            omni.generate({"prompt": prompt}, params)
            torch.cuda.synchronize()

        latencies = []
        for i in tqdm(range(num_iters),
                      desc=f"vllm-omni latency {ls['name']}", file=sys.stderr):
            params = _make_params(ls)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            omni.generate({"prompt": prompt}, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency_results.append({
            "name": ls["name"],
            "height": ls["height"],
            "width": ls["width"],
            "num_frames": ls["num_frames"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _build_throughput_scenarios(
    prompts: list[str],
    num_inference_steps: int,
    guidance_scale: float,
) -> list[dict]:
    scenarios = []
    for w in VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS:
        n = min(w.num_prompts, len(prompts))
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_frames": w.num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "prompts": prompts[:n],
        })
    return scenarios


def _build_latency_scenarios(
    prompts: list[str],
    num_inference_steps: int,
    guidance_scale: float,
) -> list[dict]:
    scenarios = []
    for w in VIDEO_DIFFUSION_LATENCY_WORKLOADS:
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_frames": w.num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "prompt": prompts[0],
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


# ---------------------------------------------------------------------------
# Correctness comparison
# ---------------------------------------------------------------------------

def _compare_latents(kb_latent_dir: str, vllm_latent_dir: str) -> dict:
    """Compare per-prompt output latents between kb-nano and vllm-omni.

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

    scenario_stats: dict[str, list[dict]] = defaultdict(list)

    for fname in common:
        kb_lat = torch.load(
            os.path.join(kb_latent_dir, fname),
            map_location="cpu", weights_only=True,
        ).detach().float().flatten()
        vllm_lat = torch.load(
            os.path.join(vllm_latent_dir, fname),
            map_location="cpu", weights_only=True,
        ).detach().float().flatten()

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

        scenario_name = fname.rsplit("_prompt", 1)[0]
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
            "num_prompts": len(batches),
            "mean_mse": float(np.mean(mses)),
            "max_mse": float(np.max(mses)),
            "mean_cosine_sim": float(np.mean(cosines)),
            "min_cosine_sim": float(np.min(cosines)),
        }
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_throughput_comparison(
    kb_results: list[dict], vllm_results: list[dict] | None,
):
    print("\n" + "=" * 90)
    print("  THROUGHPUT COMPARISON (videos/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {'Videos':>7} {'kb-nano':>12}"
    if vllm_results:
        header += f" {'vllm-omni':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = (
            f"  {kb['name']:<25} {kb['num_videos']:>7} "
            f"{kb['videos_per_second']:>12.4f}"
        )
        if vllm_results:
            vllm = next(
                (v for v in vllm_results if v["name"] == kb["name"]), None,
            )
            if vllm:
                speedup = kb["videos_per_second"] / vllm["videos_per_second"]
                line += (
                    f" {vllm['videos_per_second']:>12.4f}"
                    f" {speedup:>9.2f}x"
                )
        print(line)
    print()


def _print_latency_comparison(
    kb_results: list[dict], vllm_results: list[dict] | None,
):
    print("\n" + "=" * 90)
    print("  LATENCY COMPARISON (seconds per video)")
    print("=" * 90)
    header = (
        f"  {'Scenario':<25} {'Res':>10} {'Frames':>7}"
        f" {'kb-nano p50':>12}"
    )
    if vllm_results:
        header += f" {'vllm-omni p50':>14} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 80)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        res = f"{kb['height']}x{kb['width']}"
        line = (
            f"  {kb['name']:<25} {res:>10} {kb['num_frames']:>7}"
            f" {kb_p50:>12.3f}"
        )
        if vllm_results:
            vllm = next(
                (v for v in vllm_results if v["name"] == kb["name"]), None,
            )
            if vllm:
                vllm_lats = np.array(vllm["latencies"])
                vllm_p50 = np.percentile(vllm_lats, 50)
                speedup = vllm_p50 / kb_p50
                line += f" {vllm_p50:>14.3f} {speedup:>9.2f}x"
        print(line)
    print()


def _print_correctness_comparison(correctness: dict):
    print("\n" + "=" * 100)
    print("  CORRECTNESS COMPARISON (latent space, per-prompt)")
    print("=" * 100)
    print(
        f"  {'Scenario':<25} {'Prompts':>8} {'Mean CosSim':>12}"
        f" {'Min CosSim':>11} {'Mean MSE':>12} {'Max MSE':>12}"
        f" {'Result':>8}"
    )
    print("  " + "-" * 92)

    all_pass = True
    for scenario, stats in correctness.items():
        mean_cos = stats["mean_cosine_sim"]
        min_cos = stats["min_cosine_sim"]
        verdict = (
            "PASS" if mean_cos > 0.98
            else ("WARN" if mean_cos > 0.95 else "FAIL")
        )
        if verdict != "PASS":
            all_pass = False
        print(
            f"  {scenario:<25} {stats['num_prompts']:>8}"
            f" {mean_cos:>12.6f} {min_cos:>11.6f}"
            f" {stats['mean_mse']:>12.2e} {stats['max_mse']:>12.2e}"
            f" {verdict:>8}"
        )

    print()
    if all_pass:
        print("  All scenarios PASS (mean cosine similarity > 0.98)")
    else:
        print("  WARNING: Some scenarios have divergent outputs")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo-1.5 benchmark: kb-nano vs vllm-omni",
    )
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--skip-vllm-omni", action="store_true",
                        help="Skip vllm-omni (no correctness comparison)")
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip throughput phase (run latency only)")
    parser.add_argument("--skip-latency", action="store_true",
                        help="Skip latency phase")
    parser.add_argument("--latency-iters", type=int, default=5,
                        help="Timed iterations per latency scenario")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    gpu_name = _detect_gpu_name()

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = str(
            _PACKAGE_DIR / "tests" / "results" / gpu_name / short,
        )

    print(f"\n{'=' * 70}")
    print("  kb-nano vs vllm-omni -- HunyuanVideo-1.5 Benchmark")
    print(f"{'=' * 70}")

    bench_prompts = _get_bench_prompts(args.seed)
    print(f"  Model          : {args.model}")
    print(f"  GPU            : {gpu_name}")
    print(f"  Seed           : {args.seed}")
    print(f"  Prompts source : Movie Gen Video Bench ({len(bench_prompts)} prompts)")
    print(f"  Inference steps: {HUNYUAN_VIDEO_CONFIG.num_inference_steps}")
    print(f"  Guidance scale : {HUNYUAN_VIDEO_CONFIG.guidance_scale}")
    print(f"  Output dir     : {args.output_dir}")

    steps = HUNYUAN_VIDEO_CONFIG.num_inference_steps
    guidance = HUNYUAN_VIDEO_CONFIG.guidance_scale

    run_vllm = not args.skip_vllm_omni
    save_latents = run_vllm

    os.makedirs(args.output_dir, exist_ok=True)
    kb_latent_dir = os.path.join(args.output_dir, "latents", "kb_nano") if save_latents else None
    vllm_latent_dir = os.path.join(args.output_dir, "latents", "vllm_omni") if save_latents else None

    scenarios = (
        _build_throughput_scenarios(bench_prompts, steps, guidance)
        if not args.skip_throughput else []
    )
    latency_scenarios = (
        _build_latency_scenarios(bench_prompts, steps, guidance)
        if not args.skip_latency else []
    )

    if not args.skip_throughput:
        tp_desc = ", ".join(
            f"{s['name']}({len(s['prompts'])} prompts)" for s in scenarios
        )
        print(f"  Throughput     : {tp_desc}")
    if latency_scenarios:
        lat_desc = ", ".join(s["name"] for s in latency_scenarios)
        print(f"  Latency        : {lat_desc} ({args.latency_iters} iters)")
    print(f"{'=' * 70}")

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
    }

    # --- kb-nano benchmark ---
    kb_config = {
        **base_config,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
    }
    if kb_latent_dir:
        kb_config["latent_dir"] = kb_latent_dir
    kb_data = run_worker(
        KB_NANO_WORKER, kb_config,
        "kb-nano HunyuanVideo benchmark", timeout=36000,
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
            "vllm-omni HunyuanVideo benchmark", timeout=36000,
        )

    # --- Print results ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        vllm_tp = vllm_data.get("throughput", []) if vllm_data else None
        vllm_lat = vllm_data.get("latency", []) if vllm_data else None

        if kb_tp:
            _print_throughput_comparison(kb_tp, vllm_tp)
        if kb_lat:
            _print_latency_comparison(kb_lat, vllm_lat)

        correctness = None
        if save_latents and kb_latent_dir and vllm_latent_dir:
            correctness = _compare_latents(kb_latent_dir, vllm_latent_dir)
            if correctness:
                _print_correctness_comparison(correctness)
            else:
                print(
                    "\n  WARNING: No matching latent files found"
                    " for correctness comparison."
                )

        # Save results
        results_path = os.path.join(args.output_dir, "results.json")
        results = {
            "model": args.model,
            "gpu": gpu_name,
            "seed": args.seed,
            "prompts_source": "meta-ai-for-media-research/movie_gen_video_bench",
            "num_prompts": len(bench_prompts),
            "inference_steps": steps,
            "guidance_scale": guidance,
            "kb_nano": kb_data,
        }
        if vllm_data:
            results["vllm_omni"] = vllm_data
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
