#!/usr/bin/env python3
"""Alignment and throughput benchmark for Oasis vs official open-oasis."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_model
from torchvision.io import read_image
from torchvision.transforms.functional import pil_to_tensor, resize


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

from kb_nano.tasks.baseline.L4.oasis import (  # noqa: E402
    OasisConfig,
    OasisPipeline,
    OasisSamplingParams,
    sigmoid_beta_schedule,
)


OASIS_ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

ACTION_INDEX = {name: idx for idx, name in enumerate(OASIS_ACTION_KEYS)}
CACHE_FORMAT_VERSION = 2

LUMINE_TOKEN_TO_ACTION = {
    "E": "inventory",
    "Esc": "ESC",
    "ESC": "ESC",
    "W": "forward",
    "S": "back",
    "A": "left",
    "D": "right",
    "Space": "jump",
    "Shift": "sneak",
    "Ctrl": "sprint",
    "F": "swapHands",
    "LMB": "attack",
    "RMB": "use",
    "MMB": "pickItem",
    "Q": "drop",
}


def _load_reference_module(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from dit import DiT_models
    from vae import VAE_models

    return DiT_models, VAE_models


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


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


def _normalize_mouse_delta(value: int) -> float:
    return max(-1.0, min(1.0, float(value) / 1000.0))


def _decode_dataset_image(image_value: Any) -> torch.Tensor:
    if isinstance(image_value, bytes):
        image = Image.open(io.BytesIO(image_value)).convert("RGB")
    elif hasattr(image_value, "convert"):
        image = image_value.convert("RGB")
    else:
        raise TypeError(f"Unsupported dataset image type: {type(image_value)!r}")
    tensor = pil_to_tensor(image)
    tensor = resize(tensor, (360, 640))
    return tensor.float() / 255.0


def _parse_lumine_action(action_text: str) -> torch.Tensor:
    inner = action_text.replace("<|action_start|>", "").replace("<|action_end|>", "").strip()
    parts = [part.strip() for part in inner.split(";")]
    if len(parts) < 5:
        raise ValueError(f"Malformed action string: {action_text!r}")

    mouse_tokens = parts[0].split()
    if len(mouse_tokens) < 3:
        raise ValueError(f"Malformed mouse header: {action_text!r}")

    mouse_x = int(mouse_tokens[0])
    mouse_y = int(mouse_tokens[1])

    action = torch.zeros(len(OASIS_ACTION_KEYS), dtype=torch.float32)
    action[ACTION_INDEX["cameraX"]] = _normalize_mouse_delta(mouse_x)
    action[ACTION_INDEX["cameraY"]] = _normalize_mouse_delta(mouse_y)

    active_tokens: set[str] = set()
    for chunk in parts[1:5]:
        if not chunk:
            continue
        active_tokens.update(token for token in chunk.split() if token)

    for token in active_tokens:
        mapped = LUMINE_TOKEN_TO_ACTION.get(token)
        if mapped is not None:
            action[ACTION_INDEX[mapped]] = 1.0
            continue
        if token.isdigit() and 1 <= int(token) <= 9:
            action[ACTION_INDEX[f"hotbar.{token}"]] = 1.0

    return action


def _default_dataset_cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "oasis_cache"


def _dataset_cache_path(
    dataset_name: str,
    dataset_split: str,
    *,
    cache_dir: Path,
    num_clips: int,
    num_frames: int,
    n_prompt_frames: int,
) -> Path:
    dataset_slug = dataset_name.replace("/", "__")
    split_slug = dataset_split.replace("/", "__")
    filename = (
        f"{dataset_slug}_{split_slug}_v{CACHE_FORMAT_VERSION}_clips{num_clips}_frames{num_frames}_"
        f"prompt{n_prompt_frames}.pt"
    )
    return cache_dir / filename


def _prepare_dataset_cache(
    dataset_name: str,
    dataset_split: str,
    *,
    cache_path: Path,
    num_clips: int,
    num_frames: int,
    n_prompt_frames: int,
) -> None:
    from datasets import load_dataset

    if n_prompt_frames > num_frames:
        raise ValueError("--n-prompt-frames must be <= --num-frames")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    prompts: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    clip_metadata: list[dict[str, Any]] = []
    used_video_ids: set[str] = set()

    frame_buffer: deque[torch.Tensor] = deque()
    action_buffer: deque[torch.Tensor] = deque()
    current_video_id: str | None = None
    previous_frame_idx: int | None = None
    clip_start_frame_idx: int | None = None

    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
    for row in dataset:
        video_id = str(row["video_id"])
        frame_idx = int(row["frame_idx"])

        is_new_video = current_video_id != video_id
        is_non_consecutive = previous_frame_idx is not None and frame_idx != previous_frame_idx + 1
        if is_new_video or is_non_consecutive:
            frame_buffer.clear()
            action_buffer.clear()
            clip_start_frame_idx = None

        if video_id in used_video_ids:
            current_video_id = video_id
            previous_frame_idx = frame_idx
            frame_buffer.clear()
            action_buffer.clear()
            clip_start_frame_idx = None
            continue

        if clip_start_frame_idx is None:
            clip_start_frame_idx = frame_idx

        frame_buffer.append(_decode_dataset_image(row["image"]))
        action_buffer.append(_parse_lumine_action(row["action"]))
        current_video_id = video_id
        previous_frame_idx = frame_idx

        if len(frame_buffer) < num_frames:
            continue

        prompt = torch.stack(list(frame_buffer)[:n_prompt_frames], dim=0).half()
        if num_frames > 1:
            raw_actions = torch.stack(list(action_buffer)[: num_frames - 1], dim=0)
            zero_action = torch.zeros_like(raw_actions[:1])
            action_tensor = torch.cat([zero_action, raw_actions], dim=0)
        else:
            action_tensor = torch.zeros(1, len(OASIS_ACTION_KEYS), dtype=torch.float32)

        if action_tensor.shape[0] > 1:
            control_activity = action_tensor[1:].abs().amax(dim=0)
            distinct_controls = int((control_activity > 0).sum().item())
            camera_activity = float(
                action_tensor[1:, ACTION_INDEX["cameraX"]].abs().mean().item()
                + action_tensor[1:, ACTION_INDEX["cameraY"]].abs().mean().item()
            )
            if distinct_controls < 2 and camera_activity < 0.02:
                frame_buffer.clear()
                action_buffer.clear()
                clip_start_frame_idx = None
                continue

        prompts.append(prompt)
        actions.append(action_tensor)
        used_video_ids.add(video_id)
        clip_metadata.append(
            {
                "video_id": video_id,
                "start_frame_idx": clip_start_frame_idx,
                "end_frame_idx": frame_idx,
            }
        )

        frame_buffer.clear()
        action_buffer.clear()
        clip_start_frame_idx = None

        if len(prompts) >= num_clips:
            break

    if len(prompts) < num_clips:
        raise RuntimeError(
            f"Requested {num_clips} clips from {dataset_name}:{dataset_split}, "
            f"but only collected {len(prompts)}"
        )

    payload = {
        "dataset": dataset_name,
        "split": dataset_split,
        "num_clips": num_clips,
        "num_frames": num_frames,
        "n_prompt_frames": n_prompt_frames,
        "prompt": torch.stack(prompts, dim=0),
        "actions": torch.stack(actions, dim=0),
        "clips": clip_metadata,
    }
    torch.save(payload, cache_path)


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
    with torch.inference_mode(), _autocast(device, dtype):
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = x.reshape(bsz, params.n_prompt_frames, h // vae.patch_size, w // vae.patch_size, -1)
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    prompt_latents = x.clone()

    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, params.ddim_steps + 1, device=device)
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
            with torch.inference_mode(), _autocast(device, dtype):
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


def _benchmark(
    fn,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    units_per_iter: int = 1,
) -> dict[str, float]:
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
        "videos_per_second": units_per_iter * iters / total,
        "latency_ms_avg": total / iters * 1000.0,
        "latency_ms_p50": float(torch.tensor(times).median().item() * 1000.0),
    }


def _load_real_dataset_inputs(
    *,
    dataset_name: str,
    dataset_split: str,
    cache_dir: Path,
    num_clips: int,
    num_frames: int,
    n_prompt_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    cache_path = _dataset_cache_path(
        dataset_name,
        dataset_split,
        cache_dir=cache_dir,
        num_clips=num_clips,
        num_frames=num_frames,
        n_prompt_frames=n_prompt_frames,
    )
    if not cache_path.exists():
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--prepare-real-cache",
            "--dataset",
            dataset_name,
            "--dataset-split",
            dataset_split,
            "--dataset-cache-dir",
            str(cache_dir),
            "--num-clips",
            str(num_clips),
            "--num-frames",
            str(num_frames),
            "--n-prompt-frames",
            str(n_prompt_frames),
        ]
        subprocess.run(cmd, check=True)

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    prompt = payload["prompt"].float().to(device)
    actions = payload["actions"].float().to(device)
    info = {
        "mode": "dataset",
        "name": payload["dataset"],
        "split": payload["split"],
        "num_clips": int(payload["num_clips"]),
        "cache_path": str(cache_path),
        "clips": payload["clips"],
    }
    return prompt, actions, info


def _load_sample_inputs(
    *,
    prompt_path: Path,
    actions_path: Path,
    n_prompt_frames: int,
    num_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    prompt = _load_prompt(prompt_path, n_prompt_frames=n_prompt_frames, device=device)
    actions = _load_actions(actions_path, num_frames=num_frames, device=device)
    info = {
        "mode": "sample",
        "prompt_path": str(prompt_path),
        "actions_path": str(actions_path),
        "num_clips": 1,
    }
    return prompt, actions, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Etched/oasis-500m")
    parser.add_argument("--reference-src", default="/tmp/open-oasis")
    parser.add_argument("--prompt-path", default=None)
    parser.add_argument("--actions-path", default=None)
    parser.add_argument("--dataset", default="TESS-Computer/minecraft-vla-stage1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-cache-dir", default=str(_default_dataset_cache_dir()))
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--alignment-clips", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=4)
    parser.add_argument("--n-prompt-frames", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prepare-real-cache", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    cache_dir = Path(args.dataset_cache_dir)
    cache_path = _dataset_cache_path(
        args.dataset,
        args.dataset_split,
        cache_dir=cache_dir,
        num_clips=args.num_clips,
        num_frames=args.num_frames,
        n_prompt_frames=args.n_prompt_frames,
    )
    if args.prepare_real_cache:
        _prepare_dataset_cache(
            args.dataset,
            args.dataset_split,
            cache_path=cache_path,
            num_clips=args.num_clips,
            num_frames=args.num_frames,
            n_prompt_frames=args.n_prompt_frames,
        )
        print(f"saved real-data Oasis cache to {cache_path}", flush=True)
        import os

        os._exit(0)

    if args.n_prompt_frames > args.num_frames:
        raise ValueError("--n-prompt-frames must be <= --num-frames")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_dir = _download_model(args.model)

    use_sample_inputs = args.prompt_path is not None or args.actions_path is not None
    if use_sample_inputs:
        if args.prompt_path is None or args.actions_path is None:
            raise ValueError("Sample mode requires both --prompt-path and --actions-path")
        prompt, actions, input_info = _load_sample_inputs(
            prompt_path=Path(args.prompt_path),
            actions_path=Path(args.actions_path),
            n_prompt_frames=args.n_prompt_frames,
            num_frames=args.num_frames,
            device=device,
        )
    else:
        prompt, actions, input_info = _load_real_dataset_inputs(
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
            cache_dir=cache_dir,
            num_clips=args.num_clips,
            num_frames=args.num_frames,
            n_prompt_frames=args.n_prompt_frames,
            device=device,
        )

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
        "input": input_info,
        "config": {
            "num_frames": args.num_frames,
            "ddim_steps": args.ddim_steps,
            "n_prompt_frames": args.n_prompt_frames,
            "batch_clips": int(prompt.shape[0]),
        },
    }

    if not args.skip_alignment:
        alignment_clips = min(max(1, args.alignment_clips), int(prompt.shape[0]))
        align_prompt = prompt[:alignment_clips]
        align_actions = actions[:alignment_clips]
        with torch.inference_mode():
            ours_out = ours.rollout(align_prompt, align_actions, params, dtype=dtype)
            ref_out = _official_rollout(ref_model, ref_vae, align_prompt, align_actions, params, dtype=dtype)
        results["alignment"] = {
            "reference": "official open-oasis",
            "clips": alignment_clips,
            "prompt_latents": _tensor_metrics(ours_out.prompt_latents, ref_out["prompt_latents"]),
            "latents": _tensor_metrics(ours_out.latents, ref_out["latents"]),
            "video": _tensor_metrics(ours_out.video, ref_out["video"]),
        }

    if not args.skip_throughput:
        batch_clips = int(prompt.shape[0])
        ours_metrics = _benchmark(
            lambda: ours.rollout(prompt, actions, params, dtype=dtype),
            device=device,
            warmup=args.warmup_iters,
            iters=args.measure_iters,
            units_per_iter=batch_clips,
        )
        ref_metrics = _benchmark(
            lambda: _official_rollout(ref_model, ref_vae, prompt, actions, params, dtype=dtype),
            device=device,
            warmup=args.warmup_iters,
            iters=args.measure_iters,
            units_per_iter=batch_clips,
        )
        results["throughput"] = {
            "reference": "official open-oasis",
            "batch_clips": batch_clips,
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
