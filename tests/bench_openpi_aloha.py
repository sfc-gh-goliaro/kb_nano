"""ALOHA pen_uncap batch loading for bench_openpi (shared by parent + optional materialization).

The OpenPI subprocess often uses a different Python (``--reference-python``) where
``lerobot`` may be missing or pinned to an API incompatible with this loader.
When benchmarking vs OpenPI with real data, the parent materializes batches here
(using the kb-nano interpreter's LeRobot stack) and passes paths to the OpenPI
worker via ``aloha_batch_cache`` in the JSON config.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any

import torch

_IMAGE_TOKEN_ID = 257152
_PAD_TOKEN_ID = 0
_BOS_TOKEN_ID = 2


def _tokenize_instruction(
    text, tokenizer, num_image_tokens, num_cameras, max_length: int = 48,
):
    total_image_tokens = num_image_tokens * num_cameras
    text_enc = tokenizer.encode(text, add_special_tokens=False)
    remaining = max_length - 1 - total_image_tokens
    if remaining < 1:
        remaining = 1
    text_enc = text_enc[:remaining]
    ids = [_BOS_TOKEN_ID] + [_IMAGE_TOKEN_ID] * total_image_tokens + text_enc
    if len(ids) < max_length:
        ids = ids + [_PAD_TOKEN_ID] * (max_length - len(ids))
    ids = ids[:max_length]
    return ids


def load_aloha_dataset(
    num_samples: int,
    num_cameras: int,
    image_size: int = 224,
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    max_length: int = 48,
    device: str = "cuda",
    dtype=torch.bfloat16,
    seed: int = 42,
) -> list[dict[str, Any]] | None:
    """Load ALOHA pen_uncap observations via HuggingFace datasets."""
    import numpy as np
    from torchvision.transforms.functional import resize
    from transformers import AutoTokenizer

    rng = random.Random(seed)
    num_image_tokens = (image_size // 14) ** 2

    try:
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma2-3b-mix-224")
    except Exception:
        tokenizer = None

    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print("WARNING: datasets library not installed, falling back to synthetic data.", file=sys.stderr)
        return None

    all_camera_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    dataset_cameras = len(all_camera_keys)

    try:
        ds = hf_load_dataset(
            "physical-intelligence/aloha_pen_uncap_diverse",
            split="train",
            streaming=False,
        )
    except Exception as e:
        print(f"WARNING: Could not load ALOHA dataset: {e}", file=sys.stderr)
        return None

    frame_indices = list(range(len(ds)))
    rng.shuffle(frame_indices)
    frame_indices = frame_indices[:num_samples]

    task_text = "uncap the pen"
    batches = []
    for idx in frame_indices:
        try:
            row = ds[idx]

            cam_tensors = []
            for cam_key in all_camera_keys[:num_cameras]:
                pil_img = row[cam_key]
                img = torch.from_numpy(
                    np.array(pil_img, dtype="float32")
                ).permute(2, 0, 1) / 255.0
                img = resize(img, [image_size, image_size], antialias=True)
                cam_tensors.append(img)

            if num_cameras > dataset_cameras:
                for _ in range(num_cameras - dataset_cameras):
                    cam_tensors.append(cam_tensors[-1].clone())

            images = torch.stack(cam_tensors, dim=0)

            raw_state = row["observation.state"]
            state_vec = torch.tensor(raw_state, dtype=torch.float32)
            if state_vec.numel() < max_state_dim:
                state_vec = torch.nn.functional.pad(state_vec, (0, max_state_dim - state_vec.numel()))
            else:
                state_vec = state_vec[:max_state_dim]

            if tokenizer is not None:
                ids = _tokenize_instruction(task_text, tokenizer, num_image_tokens, num_cameras, max_length)
            else:
                total_img_tok = num_image_tokens * num_cameras
                ids = (
                    [_BOS_TOKEN_ID] + [_IMAGE_TOKEN_ID] * total_img_tok
                    + [_PAD_TOKEN_ID] * (max_length - 1 - total_img_tok)
                )
                ids = ids[:max_length]

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = (input_ids != _PAD_TOKEN_ID).long()

            raw_action = row.get("action")
            if raw_action is not None:
                gt_action = torch.tensor(raw_action, dtype=torch.float32)
            else:
                gt_action = None

            dev = torch.device(device)
            batches.append({
                "state": state_vec.unsqueeze(0).to(device=dev, dtype=dtype),
                "input_ids": input_ids,
                "pixel_values": images.unsqueeze(0).to(device=dev, dtype=dtype),
                "pixel_attention_mask": torch.ones(1, num_cameras, device=dev, dtype=torch.bool),
                "attention_mask": attention_mask,
                "num_cameras": num_cameras,
                "task_text": task_text,
                "gt_action": gt_action,
            })
        except Exception as e:
            print(f"WARNING: Skipped frame {idx}: {e}", file=sys.stderr)
            continue

    if len(batches) == 0:
        print("WARNING: No samples loaded, falling back to synthetic data.", file=sys.stderr)
        return None

    print(f"Loaded {len(batches)} ALOHA samples ({num_cameras} cameras, {image_size}px).", file=sys.stderr)
    return batches


def _batches_to_cpu_float32(batches: list[dict]) -> list[dict]:
    """Detach for cross-process reuse; OpenPI worker moves to CUDA."""
    out = []
    for b in batches:
        nb = {}
        for k, v in b.items():
            if torch.is_tensor(v):
                nb[k] = v.detach().cpu().float()
            else:
                nb[k] = v
        out.append(nb)
    return out


def materialize_aloha_cache_for_openpi(
    cache_dir: str,
    scenarios: list[dict],
    latency_scenarios: list[dict],
    *,
    seed: int,
    image_size: int,
) -> dict[str, Any] | None:
    """Load ALOHA in the parent interpreter and save batch lists for the OpenPI subprocess.

    Returns a dict suitable for ``ref_config["aloha_batch_cache"]``, or None on failure.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache: dict[str, Any] = {"throughput": {}, "latency": {}}

    for scenario in scenarios:
        name = scenario["name"]
        nc = scenario["num_cameras"]
        nreq = scenario["num_requests"]
        batches = load_aloha_dataset(
            num_samples=nreq,
            num_cameras=nc,
            image_size=image_size,
            seed=seed,
            device="cpu",
            dtype=torch.float32,
        )
        if batches is None or len(batches) < nreq:
            print(
                f"ERROR: Could not materialize ALOHA for scenario {name} (need {nreq} samples).",
                file=sys.stderr,
            )
            return None
        path = os.path.join(cache_dir, f"throughput_{name}.pt")
        torch.save(_batches_to_cpu_float32(batches[:nreq]), path)
        cache["throughput"][name] = path

    for ls in latency_scenarios:
        name = ls["name"]
        nc = ls["num_cameras"]
        nlat = ls.get("num_warmup", 3) + ls.get("num_iters", 10)
        nneed = max(nlat, 4)
        batches = load_aloha_dataset(
            num_samples=nneed,
            num_cameras=nc,
            image_size=image_size,
            seed=seed + 777,
            device="cpu",
            dtype=torch.float32,
        )
        if batches is None or len(batches) < 1:
            print(f"ERROR: Could not materialize ALOHA for latency scenario {name}.", file=sys.stderr)
            return None
        path = os.path.join(cache_dir, f"latency_{name}.pt")
        torch.save(_batches_to_cpu_float32(batches[:nneed]), path)
        cache["latency"][name] = path

    return cache
