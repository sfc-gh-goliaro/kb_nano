#!/usr/bin/env python3
"""Alignment and throughput benchmark for Oasis vs official open-oasis."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from torchvision.io import read_image
from torchvision.transforms.functional import resize


def _bootstrap_local_package() -> None:
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "kb_nano",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["kb_nano"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


_bootstrap_local_package()

from kb_nano.infra.oasis_engine import OasisEngine
from kb_nano.tasks.baseline.L4.oasis import OasisConfig, OasisPipeline, OasisSamplingParams


def _load_reference_module(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from dit import DiT_models
    from vae import VAE_models

    return DiT_models, VAE_models


def _load_prompt(prompt_path: Path, *, n_prompt_frames: int, device: torch.device) -> torch.Tensor:
    prompt = read_image(str(prompt_path))
    prompt = resize(prompt, (360, 640))
    prompt = prompt.unsqueeze(0).unsqueeze(0).float() / 255.0
    if n_prompt_frames != 1:
        prompt = prompt.repeat(1, n_prompt_frames, 1, 1, 1)
    return prompt.to(device)


def _load_actions(actions_path: Path, *, num_frames: int, device: torch.device) -> torch.Tensor:
    actions = torch.load(actions_path, weights_only=True)
    actions = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0)
    actions = actions.unsqueeze(0)[:, :num_frames]
    return actions.to(device)


def _safe_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs.reshape(-1).float()
    rhs = rhs.reshape(-1).float()
    if lhs.norm().item() == 0.0 and rhs.norm().item() == 0.0:
        return 1.0
    if lhs.norm().item() == 0.0 or rhs.norm().item() == 0.0:
        return 0.0
    return float(F.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0)).item())


def _tensor_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    diff = (lhs - rhs).abs().float()
    return {
        "cosine": _safe_cosine(lhs, rhs),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
    }


def _download_model(model_name: str) -> str:
    return snapshot_download(model_name, allow_patterns=["*.safetensors", "README.md", "LICENSE"])


def _build_reference(
    model_dir: str,
    repo_root: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    DiT_models, VAE_models = _load_reference_module(repo_root)
    model = DiT_models["DiT-S/2"]()
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    load_model(model, str(Path(model_dir) / "oasis500m.safetensors"))
    load_model(vae, str(Path(model_dir) / "vit-l-20.safetensors"))
    return model.to(device).eval(), vae.to(device).eval()


def _build_ours(
    model_dir: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OasisPipeline:
    pipeline = OasisPipeline(OasisConfig())
    pipeline.load_weights(model_dir)
    pipeline.model.to(device=device, dtype=dtype)
    pipeline.vae.to(device=device, dtype=dtype)
    return pipeline.eval()


def _official_rollout(
    model,
    vae,
    prompt: torch.Tensor,
    actions: torch.Tensor,
    params: OasisSamplingParams,
    *,
    dtype: torch.dtype,
    scaling_factor: float = 0.07843137255,
) -> dict[str, torch.Tensor]:
    device = prompt.device
    x = prompt
    bsz = x.shape[0]
    h, w = x.shape[-2:]
    x = x.reshape(bsz * params.n_prompt_frames, 3, h, w)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = x.reshape(bsz, params.n_prompt_frames, h // vae.patch_size, w // vae.patch_size, -1)
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    prompt_latents = x.clone()

    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, params.ddim_steps + 1, device=device)
    betas = torch.tensor([], device=device)
    from kb_nano.tasks.baseline.L4.oasis import sigmoid_beta_schedule

    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).view(-1, 1, 1, 1)
    generator = torch.Generator(device=device).manual_seed(params.seed or 0)

    for i in range(params.n_prompt_frames, params.num_frames):
        chunk = torch.randn((bsz, 1, *x.shape[-3:]), generator=generator, device=device)
        chunk = torch.clamp(chunk, -20, 20)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, params.ddim_steps + 1)):
            t_ctx = torch.full((bsz, i), 14, dtype=torch.long, device=device)
            t = torch.full((bsz, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((bsz, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = x[:, start_frame:].clone()
            t_curr = t[:, start_frame:]
            t_next_curr = t_next[:, start_frame:]
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                v = model(x_curr, t_curr, actions[:, start_frame : i + 1])
            x_start = alphas_cumprod[t_curr].sqrt() * x_curr - (1 - alphas_cumprod[t_curr]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t_curr]).sqrt() * x_curr - x_start) / (
                1 / alphas_cumprod[t_curr] - 1
            ).sqrt()
            alpha_next = alphas_cumprod[t_next_curr]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    latents = x
    z = latents.permute(0, 1, 3, 4, 2).reshape(bsz * params.num_frames, -1, latents.shape[2])
    with torch.inference_mode():
        video = (vae.decode(z / scaling_factor) + 1) / 2
    video = video.reshape(bsz, params.num_frames, 3, h, w)
    return {
        "prompt_latents": prompt_latents,
        "latents": latents,
        "video": video,
    }


def _benchmark(fn, *, device: torch.device, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    total = sum(times)
    return {
        "videos_per_second": iters / total,
        "latency_ms_avg": total / iters * 1000.0,
        "latency_ms_p50": float(torch.tensor(times).median().item() * 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Etched/oasis-500m")
    parser.add_argument("--reference-src", default="/tmp/open-oasis")
    parser.add_argument("--prompt-path", default="/tmp/open-oasis/sample_data/sample_image_0.png")
    parser.add_argument("--actions-path", default="/tmp/open-oasis/sample_data/sample_actions_0.one_hot_actions.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=4)
    parser.add_argument("--n-prompt-frames", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_dir = _download_model(args.model)
    prompt = _load_prompt(Path(args.prompt_path), n_prompt_frames=args.n_prompt_frames, device=device)
    actions = _load_actions(Path(args.actions_path), num_frames=args.num_frames, device=device)
    params = OasisSamplingParams(
        num_frames=args.num_frames,
        ddim_steps=args.ddim_steps,
        n_prompt_frames=args.n_prompt_frames,
        seed=0,
    )

    ours = _build_ours(model_dir, device=device, dtype=dtype)
    ref_model, ref_vae = _build_reference(model_dir, Path(args.reference_src), device=device)

    results: dict[str, Any] = {
        "seed": 0,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "model": args.model,
        "config": {
            "num_frames": args.num_frames,
            "ddim_steps": args.ddim_steps,
            "n_prompt_frames": args.n_prompt_frames,
        },
    }

    if not args.skip_alignment:
        with torch.inference_mode():
            ours_out = ours.rollout(prompt, actions, params, dtype=dtype)
            ref_out = _official_rollout(ref_model, ref_vae, prompt, actions, params, dtype=dtype)
        results["alignment"] = {
            "reference": "official open-oasis",
            "prompt_latents": _tensor_metrics(ours_out.prompt_latents, ref_out["prompt_latents"]),
            "latents": _tensor_metrics(ours_out.latents, ref_out["latents"]),
            "video": _tensor_metrics(ours_out.video, ref_out["video"]),
        }

    if not args.skip_throughput:
        ours_metrics = _benchmark(
            lambda: ours.rollout(prompt, actions, params, dtype=dtype),
            device=device,
            warmup=args.warmup_iters,
            iters=args.measure_iters,
        )
        ref_metrics = _benchmark(
            lambda: _official_rollout(ref_model, ref_vae, prompt, actions, params, dtype=dtype),
            device=device,
            warmup=args.warmup_iters,
            iters=args.measure_iters,
        )
        results["throughput"] = {
            "reference": "official open-oasis",
            "ours": ours_metrics,
            "reference_metrics": ref_metrics,
            "ratio_vs_reference": ours_metrics["videos_per_second"] / ref_metrics["videos_per_second"],
        }

    output_dir = Path(args.output_dir) if args.output_dir else Path("tests/results/H200/oasis-500m")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
