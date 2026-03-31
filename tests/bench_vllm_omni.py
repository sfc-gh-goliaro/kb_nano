#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano vs vllm-omni
for diffusion models (FLUX.1-dev, HunyuanVideo-1.5).

Runs standardized diffusion workloads and compares:
  - Throughput: images/sec or videos/sec at various resolutions
  - Latency: per-image/video latency with percentile stats
  - Correctness: per-batch/prompt MSE and cosine similarity between
    outputs of both engines

FLUX: Both engines run with output_type="latent"; packed latents are saved
per-batch and compared numerically.  Prompts from nateraw/parti-prompts.

HunyuanVideo: Since vllm-omni does not expose raw latents through its sync
API, both engines produce decoded video frames (PIL) which are saved as
numpy arrays and compared per-prompt.  Prompts from Movie Gen Video Bench.

Each engine runs in a subprocess to avoid import contamination.

Usage:
    # FLUX (default)
    python tests/bench_vllm_omni.py --model black-forest-labs/FLUX.1-dev
    python tests/bench_vllm_omni.py --skip-vllm-omni

    # HunyuanVideo
    python tests/bench_vllm_omni.py --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
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
    DIFFUSION_LATENCY_WORKLOADS,
    DIFFUSION_THROUGHPUT_WORKLOADS,
    FLUX_CONFIG,
    HUNYUAN_VIDEO_CONFIG,
    VIDEO_DIFFUSION_LATENCY_WORKLOADS,
    VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS,
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


def _is_hunyuan_video(model: str) -> bool:
    lower = model.lower()
    return "hunyuanvideo" in lower or "hunyuan-video" in lower or "hunyuan_video" in lower


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PARTI_PROMPTS: list[str] | None = None
_MOVIE_GEN_PROMPTS: list[str] | None = None


def _load_parti_prompts(seed: int = 42) -> list[str]:
    """Load all prompts from nateraw/parti-prompts (P2), deterministically shuffled."""
    from datasets import load_dataset

    ds = load_dataset("nateraw/parti-prompts", split="train")
    prompts = [row["Prompt"] for row in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def _get_parti_prompts(seed: int = 42) -> list[str]:
    global _PARTI_PROMPTS
    if _PARTI_PROMPTS is None:
        _PARTI_PROMPTS = _load_parti_prompts(seed)
    return _PARTI_PROMPTS


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


def _get_movie_gen_prompts(seed: int = 42) -> list[str]:
    global _MOVIE_GEN_PROMPTS
    if _MOVIE_GEN_PROMPTS is None:
        _MOVIE_GEN_PROMPTS = _load_movie_gen_prompts(seed)
    return _MOVIE_GEN_PROMPTS


# ═══════════════════════════════════════════════════════════════════════════
# FLUX workers
# ═══════════════════════════════════════════════════════════════════════════

FLUX_VLLM_OMNI_WORKER = r'''
import asyncio, json, os, sys, time, torch
from tqdm import tqdm

def _patch_t5_load_weights_if_needed():
    """Patch vllm-omni <= 0.18.0 T5EncoderModel.load_weights bug."""
    try:
        import vllm_omni, inspect
        if getattr(vllm_omni, "__version__", "") > "0.18.0":
            return
        from vllm_omni.diffusion.models.t5_encoder.t5_encoder import T5EncoderModel
        src = inspect.getsource(T5EncoderModel.load_weights)
        if 'name.replace(f".{weight_name}."' in src:
            return
        from collections.abc import Iterable
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        def _patched_load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            stacked_params_mapping = [
                ("qkv_proj", "q", "q"),
                ("qkv_proj", "k", "k"),
                ("qkv_proj", "v", "v"),
                ("wi", "wi_0", 0),
                ("wi", "wi_1", 1),
            ]
            params_dict = dict(self.named_parameters())
            loaded_params: set[str] = set()
            for name, loaded_weight in weights:
                original_name = name
                lookup_name = name
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if f".{weight_name}." not in name:
                        continue
                    lookup_name = name.replace(f".{weight_name}.", f".{param_name}.")
                    if lookup_name not in params_dict:
                        continue
                    param = params_dict[lookup_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                loaded_params.add(original_name)
                loaded_params.add(lookup_name)
            return loaded_params
        T5EncoderModel.load_weights = _patched_load_weights
        print("[bench] Patched vllm-omni T5EncoderModel.load_weights (v0.18.0 dotted-replace fix)", file=sys.stderr)
    except Exception as e:
        print(f"[bench] WARNING: failed to patch T5EncoderModel.load_weights: {e}", file=sys.stderr)

_patch_t5_load_weights_if_needed()

async def run_benchmark(cfg):
    from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    od_config = OmniDiffusionConfig(
        model=cfg["model"],
        dtype=torch.bfloat16,
        enforce_eager=True,
        output_type="latent",
    )
    engine = AsyncOmniDiffusion(model=cfg["model"], od_config=od_config)

    warmup_params = OmniDiffusionSamplingParams(
        height=256, width=256, num_inference_steps=2,
        guidance_scale=3.5,
    )
    warmup_params.seed = cfg["seed"]
    warmup_params.guidance_scale_provided = True
    await engine.generate_batch(["warmup"], warmup_params)

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
        params.guidance_scale_provided = True

        total_elapsed = 0.0
        total_images = 0
        desc = f"vllm-omni {scenario['name']}"
        pbar = tqdm(batches, desc=desc, unit="batch", file=sys.stderr)
        for batch_idx, batch_prompts in enumerate(pbar):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = await engine.generate_batch(batch_prompts, params)
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start
            total_elapsed += batch_elapsed
            total_images += len(batch_prompts)

            if latent_dir and output is not None:
                latent_tensor = None
                if hasattr(output, "latents") and output.latents is not None:
                    latent_tensor = output.latents
                elif hasattr(output, "images") and output.images:
                    for img in output.images:
                        if isinstance(img, torch.Tensor):
                            latent_tensor = img
                            break
                if latent_tensor is not None:
                    torch.save(
                        latent_tensor.cpu(),
                        os.path.join(latent_dir, f"{scenario['name']}_batch{batch_idx:04d}.pt"),
                    )

            pbar.set_postfix(imgs=total_images, ips=f"{total_images / total_elapsed:.2f}")

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
            height=ls["height"], width=ls["width"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 3.5),
        )
        params.seed = cfg["seed"]
        params.guidance_scale_provided = True
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for i in tqdm(range(num_warmup), desc=f"vllm-omni latency warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            await engine.generate_batch(prompts, params)
            torch.cuda.synchronize()
        latencies = []
        for i in tqdm(range(num_iters), desc=f"vllm-omni latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            await engine.generate_batch(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"], "height": ls["height"], "width": ls["width"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters, "latencies": latencies,
        })

    engine.close()
    torch.cuda.empty_cache()
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    asyncio.run(run_benchmark(cfg))

if __name__ == "__main__":
    main()
'''


FLUX_KB_NANO_WORKER = r'''
import json, os, sys, time, torch
from tqdm import tqdm

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    eng_mod = __import__(f"{pkg}.infra.diffusion_engine", fromlist=["DiffusionEngine"])
    flux_mod = __import__(f"{pkg}.tasks.baseline.L4.flux", fromlist=["DiffusionSamplingParams"])
    DiffusionEngine = eng_mod.DiffusionEngine
    DiffusionSamplingParams = flux_mod.DiffusionSamplingParams

    engine = DiffusionEngine(
        model_name=cfg["model"], seed=cfg["seed"],
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
                num_inference_steps=2, seed=cfg["seed"], output_type="latent",
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
            height=scenario["height"], width=scenario["width"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 3.5),
            seed=cfg["seed"], output_type="latent",
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
                decoded = decoded.to(dtype=pipeline.vae.dtype)
                decoded = pipeline.vae.decode(decoded, return_dict=False)[0]
                torch.save(
                    decoded.cpu(),
                    os.path.join(latent_dir, f"{scenario['name']}_batch{batch_idx:04d}.pt"),
                )
            pbar.set_postfix(imgs=total_images, ips=f"{total_images / total_elapsed:.2f}")

        all_results.append({
            "name": scenario["name"], "elapsed": total_elapsed,
            "num_images": total_images, "images_per_second": total_images / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompts"]
        params = DiffusionSamplingParams(
            height=ls["height"], width=ls["width"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 3.5),
            seed=cfg["seed"], output_type="latent",
        )
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for i in tqdm(range(num_warmup), desc=f"kb-nano latency warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize(); engine.generate(prompts, params); torch.cuda.synchronize()
        latencies = []
        for i in tqdm(range(num_iters), desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"], "height": ls["height"], "width": ls["width"],
            "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters, "latencies": latencies,
        })

    engine._cleanup()
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════════
# HunyuanVideo workers
# ═══════════════════════════════════════════════════════════════════════════

HUNYUAN_KB_NANO_WORKER = r'''
import json, os, sys, time, torch
import numpy as np
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
        model_name=cfg["model"], seed=cfg["seed"], enforce_eager=True,
    )

    frames_dir = cfg.get("frames_dir")
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    seen = set()
    for s in cfg["scenarios"] + cfg.get("latency_scenarios", []):
        key = (s["height"], s["width"], s["num_frames"])
        if key not in seen:
            seen.add(key)
            wp = HunyuanVideoDiffusionSamplingParams(
                height=s["height"], width=s["width"],
                num_frames=s["num_frames"], num_inference_steps=2,
                guidance_scale=s.get("guidance_scale", 6.0),
                seed=cfg["seed"], output_type="latent",
            )
            print(f"Warming up: {s['height']}x{s['width']} {s['num_frames']}f",
                  file=sys.stderr, flush=True)
            engine.generate(["warmup"], wp)
            torch.cuda.synchronize()

    save_frames = frames_dir is not None
    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        output_type = "pil" if save_frames else "latent"
        params = HunyuanVideoDiffusionSamplingParams(
            height=scenario["height"], width=scenario["width"],
            num_frames=scenario["num_frames"],
            num_inference_steps=scenario["num_inference_steps"],
            guidance_scale=scenario.get("guidance_scale", 6.0),
            seed=cfg["seed"], output_type=output_type,
        )
        total_elapsed = 0.0
        total_videos = 0
        desc = f"kb-nano {scenario['name']}"
        pbar = tqdm(enumerate(prompts), total=len(prompts), desc=desc, unit="vid", file=sys.stderr)
        for pi, prompt in pbar:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = engine.generate([prompt], params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_elapsed += elapsed
            total_videos += 1
            if save_frames and output.video is not None:
                frames_list = output.video
                if isinstance(frames_list, list) and frames_list:
                    arr = np.stack([np.array(f.convert("RGB")) for f in frames_list], axis=0)
                    np.save(os.path.join(frames_dir, f"{scenario['name']}_prompt{pi:04d}.npy"), arr)
            pbar.set_postfix(vids=total_videos, vps=f"{total_videos / total_elapsed:.3f}")
        all_results.append({
            "name": scenario["name"], "elapsed": total_elapsed,
            "num_videos": total_videos, "videos_per_second": total_videos / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompt = ls["prompt"]
        params = HunyuanVideoDiffusionSamplingParams(
            height=ls["height"], width=ls["width"],
            num_frames=ls["num_frames"],
            num_inference_steps=ls["num_inference_steps"],
            guidance_scale=ls.get("guidance_scale", 6.0),
            seed=cfg["seed"], output_type="latent",
        )
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for i in tqdm(range(num_warmup), desc=f"kb-nano warmup {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize(); engine.generate([prompt], params); torch.cuda.synchronize()
        latencies = []
        for i in tqdm(range(num_iters), desc=f"kb-nano latency {ls['name']}", file=sys.stderr):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate([prompt], params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"], "height": ls["height"], "width": ls["width"],
            "num_frames": ls["num_frames"], "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters, "latencies": latencies,
        })

    engine._cleanup()
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


HUNYUAN_VLLM_OMNI_WORKER = r'''
import json, os, sys, time, torch
import numpy as np
from tqdm import tqdm

def _pil_images_to_array(images):
    return np.stack([np.array(img.convert("RGB")) for img in images], axis=0)

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    seed = cfg["seed"]
    print("[vllm-omni] Creating Omni engine...", file=sys.stderr, flush=True)
    omni = Omni(model=cfg["model"], enforce_eager=True)
    print("[vllm-omni] Engine ready", file=sys.stderr, flush=True)

    frames_dir = cfg.get("frames_dir")
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    def _make_params(s):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return OmniDiffusionSamplingParams(
            height=s["height"], width=s["width"],
            num_frames=s["num_frames"],
            num_inference_steps=s["num_inference_steps"],
            guidance_scale=s.get("guidance_scale", 6.0),
            generator=generator,
        )

    all_results = []
    for scenario in cfg["scenarios"]:
        prompts = scenario["prompts"]
        total_elapsed = 0.0
        total_videos = 0
        desc = f"vllm-omni {scenario['name']}"
        pbar = tqdm(enumerate(prompts), total=len(prompts), desc=desc, unit="vid", file=sys.stderr)
        for pi, prompt in pbar:
            params = _make_params(scenario)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = omni.generate({"prompt": prompt}, params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_elapsed += elapsed
            total_videos += 1
            if frames_dir:
                items = output if isinstance(output, list) else [output]
                for item in items:
                    if hasattr(item, "images") and item.images:
                        arr = _pil_images_to_array(item.images)
                        np.save(os.path.join(frames_dir, f"{scenario['name']}_prompt{pi:04d}.npy"), arr)
            pbar.set_postfix(vids=total_videos, vps=f"{total_videos / total_elapsed:.3f}")
        all_results.append({
            "name": scenario["name"], "elapsed": total_elapsed,
            "num_videos": total_videos, "videos_per_second": total_videos / total_elapsed,
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompt = ls["prompt"]
        num_warmup = ls.get("num_warmup", 2)
        num_iters = ls.get("num_iters", 5)
        for i in tqdm(range(num_warmup), desc=f"vllm-omni warmup {ls['name']}", file=sys.stderr):
            params = _make_params(ls)
            torch.cuda.synchronize(); omni.generate({"prompt": prompt}, params); torch.cuda.synchronize()
        latencies = []
        for i in tqdm(range(num_iters), desc=f"vllm-omni latency {ls['name']}", file=sys.stderr):
            params = _make_params(ls)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            omni.generate({"prompt": prompt}, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"], "height": ls["height"], "width": ls["width"],
            "num_frames": ls["num_frames"], "num_inference_steps": ls["num_inference_steps"],
            "num_iters": num_iters, "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════════
# Scenario builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_flux_throughput_scenarios(
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
            "num_inference_steps": FLUX_CONFIG.num_inference_steps,
            "guidance_scale": FLUX_CONFIG.guidance_scale,
            "batches": batches,
            "batch_size": bs,
            "num_requests": num_requests,
        })
    return scenarios


def _build_flux_latency_scenarios(prompts: list[str]) -> list[dict]:
    scenarios = []
    for w in DIFFUSION_LATENCY_WORKLOADS:
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_inference_steps": FLUX_CONFIG.num_inference_steps,
            "guidance_scale": FLUX_CONFIG.guidance_scale,
            "prompts": prompts[:w.batch_size],
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


def _build_hunyuan_throughput_scenarios(
    prompts: list[str],
) -> list[dict]:
    steps = HUNYUAN_VIDEO_CONFIG.num_inference_steps
    guidance = HUNYUAN_VIDEO_CONFIG.guidance_scale
    scenarios = []
    for w in VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS:
        n = min(w.num_prompts, len(prompts))
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_frames": w.num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "prompts": prompts[:n],
        })
    return scenarios


def _build_hunyuan_latency_scenarios(
    prompts: list[str],
) -> list[dict]:
    steps = HUNYUAN_VIDEO_CONFIG.num_inference_steps
    guidance = HUNYUAN_VIDEO_CONFIG.guidance_scale
    scenarios = []
    for w in VIDEO_DIFFUSION_LATENCY_WORKLOADS:
        scenarios.append({
            "name": w.name,
            "height": w.height,
            "width": w.width,
            "num_frames": w.num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "prompt": prompts[0],
            "num_warmup": w.num_warmup,
            "num_iters": w.num_iters,
        })
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Correctness comparison
# ═══════════════════════════════════════════════════════════════════════════

def _compare_latents(kb_latent_dir: str, vllm_latent_dir: str) -> dict:
    """Compare per-batch output tensors (FLUX latent space)."""
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
            os.path.join(kb_latent_dir, fname), map_location="cpu", weights_only=True,
        ).detach().float().flatten()
        vllm_lat = torch.load(
            os.path.join(vllm_latent_dir, fname), map_location="cpu", weights_only=True,
        ).detach().float().flatten()
        if len(kb_lat) != len(vllm_lat):
            print(f"  WARNING: shape mismatch for {fname}, skipping", file=sys.stderr)
            continue
        kb_v, vllm_v = kb_lat.numpy(), vllm_lat.numpy()
        mse = float(np.mean((kb_v - vllm_v) ** 2))
        cos_sim = float(np.dot(kb_v, vllm_v) / (np.linalg.norm(kb_v) * np.linalg.norm(vllm_v) + 1e-12))
        scenario_name = fname.rsplit("_batch", 1)[0]
        scenario_stats[scenario_name].append({"file": fname, "mse": mse, "cosine_similarity": cos_sim})

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


def _compare_frames(kb_frames_dir: str, vllm_frames_dir: str) -> dict:
    """Compare per-prompt decoded video frames (HunyuanVideo)."""
    kb_files = sorted(
        f for f in os.listdir(kb_frames_dir) if f.endswith(".npy")
    ) if os.path.isdir(kb_frames_dir) else []
    vllm_files = sorted(
        f for f in os.listdir(vllm_frames_dir) if f.endswith(".npy")
    ) if os.path.isdir(vllm_frames_dir) else []

    common = sorted(set(kb_files) & set(vllm_files))
    if not common:
        return {}

    scenario_stats: dict[str, list[dict]] = defaultdict(list)
    for fname in common:
        kb_arr = np.load(os.path.join(kb_frames_dir, fname)).astype(np.float32).flatten()
        vllm_arr = np.load(os.path.join(vllm_frames_dir, fname)).astype(np.float32).flatten()
        if len(kb_arr) != len(vllm_arr):
            print(f"  WARNING: shape mismatch for {fname}, skipping", file=sys.stderr)
            continue
        mse = float(np.mean((kb_arr - vllm_arr) ** 2))
        psnr = float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))
        cos_sim = float(np.dot(kb_arr, vllm_arr) / (np.linalg.norm(kb_arr) * np.linalg.norm(vllm_arr) + 1e-12))
        scenario_name = fname.rsplit("_prompt", 1)[0]
        scenario_stats[scenario_name].append({
            "file": fname, "mse": mse, "psnr": psnr, "cosine_similarity": cos_sim,
        })

    results = {}
    for scenario, items in scenario_stats.items():
        mses = [b["mse"] for b in items]
        psnrs = [b["psnr"] for b in items]
        cosines = [b["cosine_similarity"] for b in items]
        results[scenario] = {
            "num_prompts": len(items),
            "mean_mse": float(np.mean(mses)),
            "max_mse": float(np.max(mses)),
            "mean_psnr": float(np.mean(psnrs)),
            "min_psnr": float(np.min(psnrs)),
            "mean_cosine_sim": float(np.mean(cosines)),
            "min_cosine_sim": float(np.min(cosines)),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Printing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _print_throughput_comparison(
    kb_results: list[dict], vllm_results: list[dict] | None, *, unit: str = "images",
):
    rate_key = "images_per_second" if unit == "images" else "videos_per_second"
    count_key = "num_images" if unit == "images" else "num_videos"
    rate_fmt = ".2f" if unit == "images" else ".4f"

    print("\n" + "=" * 90)
    print(f"  THROUGHPUT COMPARISON ({unit}/sec)")
    print("=" * 90)
    header = f"  {'Scenario':<25} {unit.title():>7} {'kb-nano':>12}"
    if vllm_results:
        header += f" {'vllm-omni':>12} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 70)

    for kb in kb_results:
        line = f"  {kb['name']:<25} {kb[count_key]:>7} {kb[rate_key]:>12{rate_fmt}}"
        if vllm_results:
            vllm = next((v for v in vllm_results if v["name"] == kb["name"]), None)
            if vllm:
                speedup = kb[rate_key] / vllm[rate_key]
                line += f" {vllm[rate_key]:>12{rate_fmt}} {speedup:>9.2f}x"
        print(line)
    print()


def _print_latency_comparison(
    kb_results: list[dict], vllm_results: list[dict] | None, *, is_video: bool = False,
):
    print("\n" + "=" * 90)
    unit_label = "seconds per video" if is_video else "seconds"
    print(f"  LATENCY COMPARISON ({unit_label})")
    print("=" * 90)
    header = f"  {'Scenario':<25}"
    if is_video:
        header += f" {'Res':>10} {'Frames':>7}"
    header += f" {'kb-nano p50':>12}"
    if vllm_results:
        header += f" {'vllm-omni p50':>14} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 80)

    for kb in kb_results:
        kb_lats = np.array(kb["latencies"])
        kb_p50 = np.percentile(kb_lats, 50)
        line = f"  {kb['name']:<25}"
        if is_video:
            res = f"{kb['height']}x{kb['width']}"
            line += f" {res:>10} {kb.get('num_frames', ''):>7}"
        line += f" {kb_p50:>12.3f}"
        if vllm_results:
            vllm = next((v for v in vllm_results if v["name"] == kb["name"]), None)
            if vllm:
                vllm_lats = np.array(vllm["latencies"])
                vllm_p50 = np.percentile(vllm_lats, 50)
                speedup = vllm_p50 / kb_p50
                line += f" {vllm_p50:>14.3f} {speedup:>9.2f}x"
        print(line)
    print()


def _print_correctness_flux(correctness: dict):
    print("\n" + "=" * 90)
    print("  CORRECTNESS COMPARISON (packed latent space, per-batch)")
    print("=" * 90)
    print(f"  {'Scenario':<25} {'Batches':>8} {'Mean CosSim':>12} {'Min CosSim':>11} {'Mean MSE':>12} {'Max MSE':>12} {'Result':>8}")
    print("  " + "-" * 88)

    all_pass = True
    for scenario, stats in correctness.items():
        mean_cos = stats["mean_cosine_sim"]
        verdict = "PASS" if mean_cos > 0.98 else ("WARN" if mean_cos > 0.95 else "FAIL")
        if verdict != "PASS":
            all_pass = False
        print(
            f"  {scenario:<25} {stats['num_batches']:>8} "
            f"{mean_cos:>12.6f} {stats['min_cosine_sim']:>11.6f} "
            f"{stats['mean_mse']:>12.2e} {stats['max_mse']:>12.2e} "
            f"{verdict:>8}"
        )
    print()
    if all_pass:
        print("  All scenarios PASS (mean cosine similarity > 0.98)")
    else:
        print("  WARNING: Some scenarios have divergent outputs")
    print()


def _print_correctness_hunyuan(correctness: dict):
    print("\n" + "=" * 110)
    print("  CORRECTNESS COMPARISON (decoded video frames, per-prompt)")
    print("=" * 110)
    print(
        f"  {'Scenario':<25} {'Prompts':>8} {'Mean CosSim':>12}"
        f" {'Min CosSim':>11} {'Mean PSNR':>10} {'Min PSNR':>9}"
        f" {'Mean MSE':>12} {'Result':>8}"
    )
    print("  " + "-" * 102)

    all_pass = True
    for scenario, stats in correctness.items():
        mean_cos = stats["mean_cosine_sim"]
        mean_psnr = stats.get("mean_psnr", 0)
        min_psnr = stats.get("min_psnr", 0)
        verdict = "PASS" if mean_cos > 0.95 else ("WARN" if mean_cos > 0.90 else "FAIL")
        if verdict != "PASS":
            all_pass = False
        print(
            f"  {scenario:<25} {stats['num_prompts']:>8}"
            f" {mean_cos:>12.6f} {stats['min_cosine_sim']:>11.6f}"
            f" {mean_psnr:>10.2f} {min_psnr:>9.2f}"
            f" {stats['mean_mse']:>12.2e}"
            f" {verdict:>8}"
        )
    print()
    if all_pass:
        print("  All scenarios PASS (mean cosine similarity > 0.95)")
    else:
        print("  WARNING: Some scenarios have divergent outputs")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diffusion benchmark: kb-nano vs vllm-omni (FLUX / HunyuanVideo)",
    )
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-vllm-omni", action="store_true",
                        help="Skip vllm-omni (no correctness comparison)")
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip throughput phase (run latency only)")
    parser.add_argument("--skip-latency", action="store_true",
                        help="Skip latency phase")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for FLUX scenarios")
    parser.add_argument("--latency-iters", type=int, default=5,
                        help="Timed iterations per latency scenario")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    gpu_name = _detect_gpu_name()
    is_video = _is_hunyuan_video(args.model)

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = str(_PACKAGE_DIR / "tests" / "results" / gpu_name / short)

    os.makedirs(args.output_dir, exist_ok=True)

    if is_video:
        _run_hunyuan(args, gpu_name)
    else:
        _run_flux(args, gpu_name)


def _run_flux(args, gpu_name: str):
    bench_prompts = _get_parti_prompts(args.seed)

    print(f"\n{'=' * 70}")
    print("  kb-nano vs vllm-omni -- FLUX Benchmark")
    print(f"{'=' * 70}")
    print(f"  Model          : {args.model}")
    print(f"  GPU            : {gpu_name}")
    print(f"  Seed           : {args.seed}")
    print(f"  Prompts source : parti-prompts ({len(bench_prompts)} prompts)")
    print(f"  Output dir     : {args.output_dir}")
    print(f"{'=' * 70}")

    run_vllm = not args.skip_vllm_omni
    save_latents = run_vllm

    kb_latent_dir = os.path.join(args.output_dir, "latents", "kb_nano") if save_latents else None
    vllm_latent_dir = os.path.join(args.output_dir, "latents", "vllm_omni") if save_latents else None

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
    }

    scenarios = _build_flux_throughput_scenarios(bench_prompts, args.batch_size) if not args.skip_throughput else []
    latency_scenarios = _build_flux_latency_scenarios(bench_prompts) if not args.skip_latency else []

    # --- kb-nano ---
    kb_config = {**base_config, "scenarios": scenarios, "latency_scenarios": latency_scenarios}
    if kb_latent_dir:
        kb_config["latent_dir"] = kb_latent_dir
    kb_data = run_worker(FLUX_KB_NANO_WORKER, kb_config, "kb-nano FLUX benchmark", timeout=36000)

    # --- vllm-omni ---
    vllm_data = None
    if run_vllm:
        vllm_config = {**base_config, "scenarios": scenarios, "latency_scenarios": latency_scenarios}
        if vllm_latent_dir:
            vllm_config["latent_dir"] = vllm_latent_dir
        vllm_data = run_worker(FLUX_VLLM_OMNI_WORKER, vllm_config, "vllm-omni FLUX benchmark", timeout=36000)

    # --- Print ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        vllm_tp = vllm_data.get("throughput", []) if vllm_data else None
        vllm_lat = vllm_data.get("latency", []) if vllm_data else None
        if kb_tp:
            _print_throughput_comparison(kb_tp, vllm_tp, unit="images")
        if kb_lat:
            _print_latency_comparison(kb_lat, vllm_lat, is_video=False)
        correctness = None
        if save_latents and kb_latent_dir and vllm_latent_dir:
            correctness = _compare_latents(kb_latent_dir, vllm_latent_dir)
            if correctness:
                _print_correctness_flux(correctness)
            else:
                print("\n  WARNING: No matching latent files found for correctness comparison.")
        _save_results(args, gpu_name, bench_prompts, kb_data, vllm_data, correctness,
                      prompts_source="nateraw/parti-prompts")
    else:
        print("ERROR: kb-nano benchmark failed.")
        sys.exit(1)


def _run_hunyuan(args, gpu_name: str):
    bench_prompts = _get_movie_gen_prompts(args.seed)

    print(f"\n{'=' * 70}")
    print("  kb-nano vs vllm-omni -- HunyuanVideo-1.5 Benchmark")
    print(f"{'=' * 70}")
    print(f"  Model          : {args.model}")
    print(f"  GPU            : {gpu_name}")
    print(f"  Seed           : {args.seed}")
    print(f"  Prompts source : Movie Gen Video Bench ({len(bench_prompts)} prompts)")
    print(f"  Inference steps: {HUNYUAN_VIDEO_CONFIG.num_inference_steps}")
    print(f"  Guidance scale : {HUNYUAN_VIDEO_CONFIG.guidance_scale}")
    print(f"  Output dir     : {args.output_dir}")

    run_vllm = not args.skip_vllm_omni
    save_frames = run_vllm

    kb_frames_dir = os.path.join(args.output_dir, "frames", "kb_nano") if save_frames else None
    vllm_frames_dir = os.path.join(args.output_dir, "frames", "vllm_omni") if save_frames else None

    base_config = {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": True,
        "project_root": str(_PROJECT_ROOT),
        "package_name": "kb_nano",
    }

    scenarios = _build_hunyuan_throughput_scenarios(bench_prompts) if not args.skip_throughput else []
    latency_scenarios = _build_hunyuan_latency_scenarios(bench_prompts) if not args.skip_latency else []

    if not args.skip_throughput:
        tp_desc = ", ".join(f"{s['name']}({len(s['prompts'])} prompts)" for s in scenarios)
        print(f"  Throughput     : {tp_desc}")
    if latency_scenarios:
        lat_desc = ", ".join(s["name"] for s in latency_scenarios)
        print(f"  Latency        : {lat_desc} ({args.latency_iters} iters)")
    print(f"{'=' * 70}")

    # --- kb-nano ---
    kb_config = {**base_config, "scenarios": scenarios, "latency_scenarios": latency_scenarios}
    if kb_frames_dir:
        kb_config["frames_dir"] = kb_frames_dir
    kb_data = run_worker(HUNYUAN_KB_NANO_WORKER, kb_config, "kb-nano HunyuanVideo benchmark", timeout=36000)

    # --- vllm-omni ---
    vllm_data = None
    if run_vllm:
        vllm_config = {**base_config, "scenarios": scenarios, "latency_scenarios": latency_scenarios}
        if vllm_frames_dir:
            vllm_config["frames_dir"] = vllm_frames_dir
        vllm_data = run_worker(HUNYUAN_VLLM_OMNI_WORKER, vllm_config, "vllm-omni HunyuanVideo benchmark", timeout=36000)

    # --- Print ---
    if kb_data:
        kb_tp = kb_data.get("throughput", [])
        kb_lat = kb_data.get("latency", [])
        vllm_tp = vllm_data.get("throughput", []) if vllm_data else None
        vllm_lat = vllm_data.get("latency", []) if vllm_data else None
        if kb_tp:
            _print_throughput_comparison(kb_tp, vllm_tp, unit="videos")
        if kb_lat:
            _print_latency_comparison(kb_lat, vllm_lat, is_video=True)
        correctness = None
        if save_frames and kb_frames_dir and vllm_frames_dir:
            correctness = _compare_frames(kb_frames_dir, vllm_frames_dir)
            if correctness:
                _print_correctness_hunyuan(correctness)
            else:
                print("\n  WARNING: No matching frame files found for correctness comparison.")
        _save_results(args, gpu_name, bench_prompts, kb_data, vllm_data, correctness,
                      prompts_source="meta-ai-for-media-research/movie_gen_video_bench")
    else:
        print("ERROR: kb-nano benchmark failed.")
        sys.exit(1)


def _save_results(
    args, gpu_name: str, bench_prompts: list[str],
    kb_data: dict, vllm_data: dict | None, correctness: dict | None,
    prompts_source: str,
):
    results_path = os.path.join(args.output_dir, "results.json")
    results = {
        "model": args.model,
        "gpu": gpu_name,
        "seed": args.seed,
        "prompts_source": prompts_source,
        "num_prompts": len(bench_prompts),
        "kb_nano": kb_data,
    }
    if vllm_data:
        results["vllm_omni"] = vllm_data
    if correctness:
        results["correctness"] = correctness
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
