#!/usr/bin/env python3
"""
Smoke test & correctness benchmark: kb-nano HunyuanVideo-1.5 vs vllm-omni.

Exercises the full pipeline (text encoding -> diffusion -> optional VAE decode)
with minimal resolution and steps to catch errors quickly.

Usage:
    # kb-nano only (smoke test):
    python tests/test_hunyuan_video.py --skip-vllm-omni

    # Full comparison:
    python tests/test_hunyuan_video.py

    # Custom model repo:
    python tests/test_hunyuan_video.py --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PACKAGE_DIR))

from kb_nano.bench.utils.worker import run_worker

MODEL = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

TEST_PROMPTS = [
    "A golden retriever playing in the snow, cinematic lighting",
    "A serene lakeside sunrise with mist over the water",
]

LARGE_PROMPTS = [
    "A golden retriever playing in the snow, cinematic lighting",
    "A serene lakeside sunrise with mist over the water",
    "Time-lapse of a flower blooming in a garden, macro photography",
    "A bustling Tokyo street at night with neon signs and rain reflections",
    "An astronaut floating in zero gravity inside a space station",
    "Ocean waves crashing against a rocky cliff during a storm, slow motion",
    "A chef preparing sushi in a traditional Japanese restaurant, close-up",
    "Northern lights dancing over a snowy mountain landscape in Iceland",
    "A hummingbird hovering near a bright red flower, high frame rate",
    "A classic car driving through a desert highway at sunset",
    "Underwater footage of a coral reef with tropical fish swimming by",
    "A cat stalking a laser dot across a hardwood floor, playful motion",
    "Steam rising from a cup of coffee on a rainy windowsill morning",
    "Fireworks exploding over a city skyline on New Year's Eve",
    "A ballet dancer performing a pirouette on an empty stage, dramatic lighting",
    "A drone shot ascending from a dense forest canopy revealing a mountain range",
]

SMALL_VIDEO = dict(height=192, width=320, num_frames=9, num_inference_steps=4, guidance_scale=6.0)
MEDIUM_VIDEO = dict(height=480, width=832, num_frames=25, num_inference_steps=10, guidance_scale=6.0)
LARGE_VIDEO = dict(height=480, width=832, num_frames=49, num_inference_steps=30, guidance_scale=6.0)

# ---------------------------------------------------------------------------
# kb-nano worker
# ---------------------------------------------------------------------------
KB_NANO_WORKER = r'''
import json, os, sys, time, torch, traceback
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

    all_results = []
    for scenario in cfg["scenarios"]:
        params = HunyuanVideoDiffusionSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_frames=scenario["num_frames"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 6.0),
            seed=cfg["seed"],
            output_type="latent",
        )

        prompts = scenario["prompts"]
        print(f"\n[kb-nano] Running scenario: {scenario['name']}", file=sys.stderr, flush=True)
        print(f"  {len(prompts)} prompts, {scenario['height']}x{scenario['width']}, "
              f"{scenario['num_frames']} frames, {scenario['num_inference_steps']} steps",
              file=sys.stderr, flush=True)

        # Warmup run to ensure model is loaded and CUDA is ready
        warmup_params = HunyuanVideoDiffusionSamplingParams(
            height=scenario["height"],
            width=scenario["width"],
            num_frames=scenario["num_frames"],
            num_inference_steps=2,
            guidance_scale=scenario.get("guidance_scale", 6.0),
            seed=cfg["seed"],
            output_type="latent",
        )
        print("  Warming up...", file=sys.stderr, flush=True)
        engine.generate(["warmup"], warmup_params)
        torch.cuda.synchronize()
        print("  Warmup done", file=sys.stderr, flush=True)

        all_latents = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for pi, prompt in enumerate(prompts):
            output = engine.generate([prompt], params)
            if output.latents is not None:
                all_latents.append(output.latents.cpu())
            print(f"  [{pi+1}/{len(prompts)}] done", file=sys.stderr, flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"  Completed in {elapsed:.2f}s ({elapsed/len(prompts):.2f}s/prompt)", file=sys.stderr, flush=True)

        latent_shape = None
        if latent_dir and all_latents:
            stacked = torch.cat(all_latents, dim=0)
            latent_shape = list(stacked.shape)
            torch.save(stacked, os.path.join(latent_dir, f"{scenario['name']}.pt"))
            print(f"  Saved latents: shape={latent_shape}, "
                  f"dtype={stacked.dtype}", file=sys.stderr, flush=True)

        all_results.append({
            "name": scenario["name"],
            "elapsed": elapsed,
            "num_prompts": len(prompts),
            "latent_shape": latent_shape,
        })

    engine._cleanup()

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": all_results}, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# vllm-omni worker
# ---------------------------------------------------------------------------
VLLM_OMNI_WORKER = r'''
import json, os, sys, time, torch, numpy as np

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    model = cfg["model"]
    seed = cfg["seed"]

    print("[vllm-omni] Creating Omni engine...", file=sys.stderr, flush=True)
    omni = Omni(model=model, enforce_eager=True)
    print("[vllm-omni] Engine ready", file=sys.stderr, flush=True)

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        print(f"\n[vllm-omni] Running scenario: {scenario['name']}", file=sys.stderr, flush=True)
        print(f"  {len(prompts)} prompts, {scenario['height']}x{scenario['width']}, "
              f"{scenario['num_frames']} frames, {scenario['num_inference_steps']} steps",
              file=sys.stderr, flush=True)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for pi, prompt in enumerate(prompts):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            params = OmniDiffusionSamplingParams(
                height=scenario["height"],
                width=scenario["width"],
                num_frames=scenario["num_frames"],
                num_inference_steps=scenario["num_inference_steps"],
                guidance_scale=scenario.get("guidance_scale", 6.0),
                generator=generator,
            )
            output = omni.generate({"prompt": prompt}, params)
            print(f"  [{pi+1}/{len(prompts)}] done", file=sys.stderr, flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"  Completed in {elapsed:.2f}s ({elapsed/len(prompts):.2f}s/prompt)", file=sys.stderr, flush=True)

        all_results.append({
            "name": scenario["name"],
            "elapsed": elapsed,
            "num_prompts": len(prompts),
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": all_results}, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# HF diffusers reference worker (for correctness comparison)
# ---------------------------------------------------------------------------
HF_DIFFUSERS_WORKER = r'''
import json, os, sys, time, torch, numpy as np

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from diffusers import HunyuanVideo15Pipeline as HFPipeline
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from diffusers.utils.torch_utils import randn_tensor

    model = cfg["model"]
    seed = cfg["seed"]
    device = torch.device("cuda")

    print("[diffusers] Loading pipeline...", file=sys.stderr, flush=True)
    pipe = HFPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.vae.to(torch.float32)
    print("[diffusers] Pipeline ready", file=sys.stderr, flush=True)

    latent_dir = cfg.get("latent_dir")
    if latent_dir:
        os.makedirs(latent_dir, exist_ok=True)

    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        print(f"\n[diffusers] Running scenario: {scenario['name']}", file=sys.stderr, flush=True)

        all_latents = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for pi, prompt in enumerate(prompts):
            generator = torch.Generator(device=device).manual_seed(seed)
            sigmas = np.linspace(1.0, 0.0, scenario["num_inference_steps"] + 1)[:-1].tolist()
            output = pipe(
                prompt=prompt,
                height=scenario["height"],
                width=scenario["width"],
                num_frames=scenario["num_frames"],
                num_inference_steps=scenario["num_inference_steps"],
                sigmas=sigmas,
                generator=generator,
                output_type="latent",
            )
            if hasattr(output, "frames") and output.frames is not None:
                latent = output.frames
            elif hasattr(output, "latents") and output.latents is not None:
                latent = output.latents
            else:
                latent = None
            if latent is not None:
                if isinstance(latent, torch.Tensor):
                    all_latents.append(latent.cpu())
                    print(f"  [{pi+1}/{len(prompts)}] done, latent shape={list(latent.shape)}",
                          file=sys.stderr, flush=True)
                else:
                    print(f"  [{pi+1}/{len(prompts)}] done, latent type={type(latent)}",
                          file=sys.stderr, flush=True)
            else:
                print(f"  [{pi+1}/{len(prompts)}] done, no latent found", file=sys.stderr, flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"  Completed in {elapsed:.2f}s ({elapsed/len(prompts):.2f}s/prompt)", file=sys.stderr, flush=True)

        if latent_dir and all_latents:
            stacked = torch.cat(all_latents, dim=0)
            torch.save(stacked, os.path.join(latent_dir, f"{scenario['name']}.pt"))

        all_results.append({
            "name": scenario["name"],
            "elapsed": elapsed,
            "num_prompts": len(prompts),
        })

    del pipe
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": all_results}, f)

if __name__ == "__main__":
    main()
'''


def _compare_latents(kb_dir: str, vllm_dir: str) -> dict:
    """Compare latent tensors between kb-nano and vllm-omni."""
    import torch

    kb_files = sorted(f for f in os.listdir(kb_dir) if f.endswith(".pt")) if os.path.isdir(kb_dir) else []
    vllm_files = sorted(f for f in os.listdir(vllm_dir) if f.endswith(".pt")) if os.path.isdir(vllm_dir) else []

    common = sorted(set(kb_files) & set(vllm_files))
    if not common:
        return {}

    results = {}
    for fname in common:
        kb_lat = torch.load(os.path.join(kb_dir, fname), map_location="cpu", weights_only=True).float().flatten()
        vllm_lat = torch.load(os.path.join(vllm_dir, fname), map_location="cpu", weights_only=True).float().flatten()

        if len(kb_lat) != len(vllm_lat):
            print(f"  Shape mismatch for {fname}: kb={kb_lat.shape} vs vllm={vllm_lat.shape}")
            results[fname] = {"error": "shape_mismatch", "kb_shape": list(kb_lat.shape), "vllm_shape": list(vllm_lat.shape)}
            continue

        kb_v, vllm_v = kb_lat.numpy(), vllm_lat.numpy()
        mse = float(np.mean((kb_v - vllm_v) ** 2))
        cos_sim = float(np.dot(kb_v, vllm_v) / (np.linalg.norm(kb_v) * np.linalg.norm(vllm_v) + 1e-12))
        max_abs_diff = float(np.max(np.abs(kb_v - vllm_v)))

        results[fname] = {"mse": mse, "cosine_similarity": cos_sim, "max_abs_diff": max_abs_diff}

    return results


def main():
    parser = argparse.ArgumentParser(description="HunyuanVideo-1.5 test: kb-nano vs vllm-omni")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-vllm-omni", action="store_true")
    parser.add_argument("--skip-diffusers", action="store_true",
                        help="Skip HF diffusers reference (no correctness comparison)")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small",
                        help="Video size preset (small=192x320x9x4, medium=480x832x25x10, large=480x832x49x30)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    video_cfg = {"small": SMALL_VIDEO, "medium": MEDIUM_VIDEO, "large": LARGE_VIDEO}[args.size]

    if args.output_dir is None:
        args.output_dir = str(_THIS_DIR / "results" / "hunyuan_video_test")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nHunyuanVideo-1.5 Benchmark ({args.size})")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Video config: {video_cfg}")
    print(f"Prompts: {len(prompts_for_run)}")
    print(f"Output dir: {args.output_dir}")

    prompts_for_run = LARGE_PROMPTS if args.size == "large" else TEST_PROMPTS
    scenarios = [
        {
            "name": f"hunyuan_{args.size}",
            "prompts": prompts_for_run,
            **video_cfg,
        },
    ]

    run_vllm = not args.skip_vllm_omni
    kb_latent_dir = os.path.join(args.output_dir, "latents", "kb_nano")
    vllm_latent_dir = os.path.join(args.output_dir, "latents", "vllm_omni")

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "project_root": str(_PACKAGE_DIR),
        "package_name": "kb_nano",
        "scenarios": scenarios,
    }

    # --- kb-nano ---
    print("\n" + "=" * 60)
    print("  kb-nano benchmark")
    print("=" * 60)

    kb_config = {**base_config, "latent_dir": kb_latent_dir}
    kb_data = run_worker(KB_NANO_WORKER, kb_config, "kb-nano HunyuanVideo", timeout=3600)

    if kb_data:
        for r in kb_data["results"]:
            print(f"  {r['name']}: {r['elapsed']:.2f}s, latent_shape={r.get('latent_shape')}")
    else:
        print("  ERROR: kb-nano benchmark failed!")
        sys.exit(1)

    # --- vllm-omni ---
    vllm_data = None
    if run_vllm:
        print("\n" + "=" * 60)
        print("  vllm-omni benchmark")
        print("=" * 60)

        vllm_config = {**base_config, "latent_dir": vllm_latent_dir}
        vllm_data = run_worker(VLLM_OMNI_WORKER, vllm_config, "vllm-omni HunyuanVideo", timeout=3600)

        if vllm_data:
            for r in vllm_data["results"]:
                print(f"  {r['name']}: {r['elapsed']:.2f}s")
        else:
            print("  WARNING: vllm-omni benchmark failed")

    # --- HF diffusers reference ---
    diffusers_data = None
    run_diffusers = not args.skip_diffusers
    diffusers_latent_dir = os.path.join(args.output_dir, "latents", "diffusers")
    if run_diffusers:
        print("\n" + "=" * 60)
        print("  HF diffusers reference")
        print("=" * 60)

        diffusers_config = {**base_config, "latent_dir": diffusers_latent_dir}
        diffusers_data = run_worker(HF_DIFFUSERS_WORKER, diffusers_config,
                                    "HF diffusers HunyuanVideo", timeout=3600)

        if diffusers_data:
            for r in diffusers_data["results"]:
                print(f"  {r['name']}: {r['elapsed']:.2f}s")
        else:
            print("  WARNING: diffusers reference failed")

    # --- Correctness: kb-nano vs diffusers ---
    if run_diffusers and diffusers_data:
        print("\n" + "=" * 60)
        print("  CORRECTNESS: kb-nano vs HF diffusers")
        print("=" * 60)

        correctness_diffusers = _compare_latents(kb_latent_dir, diffusers_latent_dir)
        if correctness_diffusers:
            for name, stats in correctness_diffusers.items():
                if "error" in stats:
                    print(f"  {name}: {stats['error']} (kb={stats.get('kb_shape')}, ref={stats.get('vllm_shape')})")
                else:
                    cos = stats["cosine_similarity"]
                    verdict = "PASS" if cos > 0.99 else ("WARN" if cos > 0.95 else "FAIL")
                    print(f"  {name}: CosSim={cos:.6f}  MSE={stats['mse']:.2e}  "
                          f"MaxAbsDiff={stats['max_abs_diff']:.2e}  [{verdict}]")
        else:
            print("  No matching latent files found for comparison.")

    # --- Comparison ---
    if run_vllm and vllm_data:
        print("\n" + "=" * 60)
        print("  CORRECTNESS COMPARISON")
        print("=" * 60)

        correctness = _compare_latents(kb_latent_dir, vllm_latent_dir)
        if correctness:
            for name, stats in correctness.items():
                if "error" in stats:
                    print(f"  {name}: {stats['error']}")
                else:
                    cos = stats["cosine_similarity"]
                    verdict = "PASS" if cos > 0.99 else ("WARN" if cos > 0.95 else "FAIL")
                    print(f"  {name}: CosSim={cos:.6f}  MSE={stats['mse']:.2e}  "
                          f"MaxAbsDiff={stats['max_abs_diff']:.2e}  [{verdict}]")
        else:
            print("  No matching latent files found for comparison.")

    # --- Performance comparison ---
    if run_vllm and vllm_data and kb_data:
        print("\n" + "=" * 60)
        print("  PERFORMANCE COMPARISON")
        print("=" * 60)
        for kb_r in kb_data["results"]:
            vllm_r = next((v for v in vllm_data["results"] if v["name"] == kb_r["name"]), None)
            if vllm_r:
                speedup = vllm_r["elapsed"] / kb_r["elapsed"]
                print(f"  {kb_r['name']}: kb-nano={kb_r['elapsed']:.2f}s  "
                      f"vllm-omni={vllm_r['elapsed']:.2f}s  speedup={speedup:.2f}x")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    results = {"model": args.model, "seed": args.seed, "video_config": video_cfg,
               "kb_nano": kb_data}
    if vllm_data:
        results["vllm_omni"] = vllm_data
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
