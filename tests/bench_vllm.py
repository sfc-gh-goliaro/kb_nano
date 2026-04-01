#!/usr/bin/env python3
"""
Throughput and alignment benchmark: kb-nano baseline vs vLLM.

For LLM models: runs three text-only scenarios (prefill-heavy, balanced,
decode-heavy) with random token IDs.

For VLM models (Qwen2-VL, Qwen3-VL): runs three throughput scenarios
(text-only, image, video) and two latency scenarios (single-image,
single-video) using real multimodal datasets (VisionArena, MMVU).

Each engine (vLLM, kb-nano) is loaded once in a single long-lived subprocess
that processes all scenarios sequentially, avoiding repeated model loading.

Usage:
    # LLM benchmark
    python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct

    # VLM benchmark (auto-detected from model name)
    python tests/bench_vllm.py --model Qwen/Qwen2-VL-7B-Instruct

    python tests/bench_vllm.py --skip-vllm  # kb-nano only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from random import randint

import subprocess

import numpy as np


def _detect_gpu_name() -> str:
    """Return short GPU name (e.g. 'H200', 'B200') via nvidia-smi."""
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

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from kb_nano.bench.utils.worker import run_worker


SCENARIOS = [
    {"name": "prefill-heavy", "input_len": 1024, "output_len": 512},
    {"name": "balanced",      "input_len": 512,  "output_len": 512},
    {"name": "decode-heavy",  "input_len": 512,  "output_len": 1024},
]

LATENCY_SCENARIOS = [
    {"name": "single-request",  "input_len": 128, "output_len": 128, "batch_size": 1},
    {"name": "fixed-batch-32",  "input_len": 128, "output_len": 128, "batch_size": 32},
]

VLM_SCENARIOS = [
    {"name": "text-only",  "modality": "text",  "input_len": 512, "output_len": 1024},
    {"name": "image",      "modality": "image", "output_len": 512,
     "dataset": "lmarena-ai/VisionArena-Chat", "dataset_split": "train"},
    {"name": "video",      "modality": "video", "output_len": 512,
     "dataset": "yale-nlp/MMVU", "dataset_split": "validation"},
]

WHISPER_SCENARIOS = [
    {"name": "librispeech", "output_len": 448,
     "dataset": "openslr/librispeech_asr", "dataset_split": "test.clean",
     "use_full_dataset": True},
]

WHISPER_LATENCY_SCENARIOS = [
    {"name": "single-utterance", "output_len": 448, "batch_size": 1,
     "dataset": "openslr/librispeech_asr", "dataset_split": "test.clean"},
    {"name": "fixed-batch-32", "output_len": 448, "batch_size": 32,
     "dataset": "openslr/librispeech_asr", "dataset_split": "test.clean"},
]

VLM_LATENCY_SCENARIOS = [
    {"name": "single-image", "modality": "image", "output_len": 128, "batch_size": 1,
     "dataset": "lmarena-ai/VisionArena-Chat", "dataset_split": "train"},
    {"name": "single-video", "modality": "video", "output_len": 128, "batch_size": 1,
     "dataset": "yale-nlp/MMVU", "dataset_split": "validation"},
]


def _is_vlm_model(model_name: str) -> bool:
    lower = model_name.lower()
    return "qwen" in lower and "vl" in lower


def _is_whisper_model(model_name: str) -> bool:
    lower = model_name.lower()
    return "whisper" in lower


# ---------------------------------------------------------------------------
# Multi-scenario vLLM subprocess worker (LLM, text-only)
# ---------------------------------------------------------------------------
VLLM_WORKER = r'''
import json, os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

def main():
    from vllm import LLM, SamplingParams

    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    llm_kwargs = dict(
        model=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
    )
    if cfg.get("load_format"):
        llm_kwargs["load_format"] = cfg["load_format"]
    llm = LLM(**llm_kwargs)

    # Warmup
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=16),
    )

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        prompt_token_ids = scenario["prompt_token_ids"]
        output_lens = scenario["output_lens"]
        temperature = cfg.get("temperature", 0.0)

        sp_list = [
            SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=ol)
            for ol in output_lens
        ]

        vllm_prompts = [dict(prompt_token_ids=p) for p in prompt_token_ids]
        start = time.perf_counter()
        outputs = llm.generate(vllm_prompts, sp_list)
        elapsed = time.perf_counter() - start

        total_prompt_tokens = sum(
            len(o.prompt_token_ids) if o.prompt_token_ids else 0
            for o in outputs
        )
        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs if c)
            for o in outputs
        )

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {
                    "text": o.outputs[0].text,
                    "token_ids": list(o.outputs[0].token_ids),
                }
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = [dict(prompt_token_ids=p) for p in ls["prompt_token_ids"]]
        sp = SamplingParams(temperature=0.0,
                            ignore_eos=True, max_tokens=ls["output_len"])
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            llm.generate(prompts, sp, use_tqdm=False)
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            llm.generate(prompts, sp, use_tqdm=False)
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "input_len": ls["input_len"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del llm

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Multi-scenario kb-nano subprocess worker
# ---------------------------------------------------------------------------
KB_NANO_WORKER = r'''
import json, sys, time

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    engine = LlamaEngine(**engine_kwargs)

    # Warmup
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    import torch
    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        prompts = scenario["prompt_token_ids"]
        output_lens = scenario["output_lens"]
        temperature = cfg.get("temperature", 0.0)
        top_p = cfg.get("top_p", 1.0)

        sp_list = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=ol,
                ignore_eos=True,
            )
            for ol in output_lens
        ]

        engine.block_manager.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(prompts, sp_list, use_tqdm=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_input_tokens = sum(len(p) for p in prompts)
        total_output_tokens = sum(len(o.token_ids) for o in outputs)

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {
                    "generated_text": o.generated_text,
                    "token_ids": o.token_ids,
                }
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompt_token_ids"]
        sp = SamplingParams(temperature=0.0,
                            ignore_eos=True, max_tokens=ls["output_len"])
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "input_len": ls["input_len"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

    del engine

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Shared multimodal data loading (no vLLM imports -- cv2, numpy, PIL only)
# Inlined into both VLM workers so each subprocess is self-contained.
# ---------------------------------------------------------------------------
_MM_PRELOAD_FN = r'''
import math
import numpy as np
from io import BytesIO
from PIL import Image
from tqdm import tqdm

def _load_video_opencv(video_path, num_frames=32):
    """Load video frames with OpenCV, matching vLLM's OpenCVVideoBackend."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames_num / original_fps if original_fps > 0 else 0

    num_frames_to_sample = total_frames_num
    if num_frames > 0:
        num_frames_to_sample = min(num_frames, total_frames_num)
    num_frames_to_sample = max(1, num_frames_to_sample)

    if num_frames_to_sample == total_frames_num:
        frame_idx = list(range(num_frames_to_sample))
    else:
        frame_idx = np.linspace(
            0, total_frames_num - 1, num_frames_to_sample, dtype=int
        ).tolist()

    frame_idx_set = set(frame_idx)
    max_idx = max(frame_idx)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.empty((num_frames_to_sample, height, width, 3), dtype=np.uint8)

    i = 0
    valid_frame_indices = []
    for idx in range(max_idx + 1):
        ok = cap.grab()
        if not ok:
            continue
        if idx in frame_idx_set:
            ret, frame = cap.retrieve()
            if ret:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                valid_frame_indices.append(idx)
                i += 1

    cap.release()
    valid_num_frames = len(valid_frame_indices)
    frames = frames[:valid_num_frames]

    metadata = {
        "total_num_frames": total_frames_num,
        "fps": original_fps,
        "duration": duration,
        "video_backend": "opencv",
        "frames_indices": valid_frame_indices,
        "do_sample_frames": valid_num_frames == total_frames_num,
    }
    return frames, metadata


def _preload_mm_data(dataset_name, dataset_split, num_seqs, seed,
                     num_video_frames=32):
    """Pre-download and load all images/videos into memory.

    Returns list of dicts with keys:
      - prompt: str
      - images: list[PIL.Image] or None
      - video_frames: np.ndarray (T,H,W,3) or None
      - video_metadata: dict or None
    """
    from datasets import load_dataset
    use_streaming = "MMVU" not in dataset_name
    data = load_dataset(dataset_name, split=dataset_split,
                        streaming=use_streaming)
    data = data.shuffle(seed=seed)

    results = []
    if "VisionArena" in dataset_name:
        pbar = tqdm(data, total=num_seqs, desc="Loading images")
        for item in pbar:
            if len(results) >= num_seqs:
                break
            try:
                prompt = item["conversation"][0][0]["content"]
                if "base64" in prompt or len(prompt) > 4096:
                    continue
                img = item["images"][0]
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(BytesIO(img["bytes"]))
                if not isinstance(img, Image.Image):
                    continue
                img = img.convert("RGB")
                w, h = img.size
                if w * h > 2048 * 2048:
                    continue
            except Exception:
                continue
            results.append({
                "prompt": prompt,
                "images": [img],
                "video_frames": None,
                "video_metadata": None,
            })
            pbar.update(0)
        pbar.close()
    elif "MMVU" in dataset_name:
        from huggingface_hub import snapshot_download
        local_root = snapshot_download(dataset_name, repo_type="dataset")
        remote_root = (
            f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
        )
        pbar = tqdm(data, total=num_seqs, desc="Loading videos")
        for item in pbar:
            if len(results) >= num_seqs:
                break
            prompt = item["question"] + " " + " ".join(
                f"{k}.{v}" for k, v in item["choices"].items())
            video_path = item["video"].replace(remote_root, local_root)
            frames, metadata = _load_video_opencv(
                video_path, num_frames=num_video_frames)
            results.append({
                "prompt": prompt,
                "images": None,
                "video_frames": frames,
                "video_metadata": metadata,
            })
            pbar.update(0)
        pbar.close()
    return results


def _filter_and_prepare(mm_data, processor, max_input_tokens):
    """Filter items by token count and pre-compute chat text in one pass."""
    prepared = []
    for item in tqdm(mm_data, desc="Filtering & preparing prompts"):
        try:
            messages = [{"role": "user", "content": []}]
            images_for_proc = None
            videos_for_proc = None
            if item["images"] is not None:
                for img in item["images"]:
                    messages[0]["content"].append(
                        {"type": "image", "image": img})
                images_for_proc = item["images"]
            if item["video_frames"] is not None:
                frames_pil = [
                    Image.fromarray(item["video_frames"][j]).convert("RGB")
                    for j in range(item["video_frames"].shape[0])
                ]
                messages[0]["content"].append(
                    {"type": "video", "video": frames_pil})
                videos_for_proc = [frames_pil]
            messages[0]["content"].append(
                {"type": "text", "text": item["prompt"]})
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=images_for_proc,
                videos=videos_for_proc,
                return_tensors="pt",
                padding=True,
            )
            num_tokens = inputs["input_ids"].shape[1]
            if num_tokens <= max_input_tokens:
                item["chat_text"] = text
                prepared.append(item)
        except Exception:
            continue
    return prepared
'''

# ---------------------------------------------------------------------------
# Multi-scenario vLLM subprocess worker (VLM, multi-modal)
# ---------------------------------------------------------------------------
VLLM_VLM_WORKER = _MM_PRELOAD_FN + r'''
import json, os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    llm_kwargs = dict(
        model=model_name,
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
    )
    if cfg.get("load_format"):
        llm_kwargs["load_format"] = cfg["load_format"]
    llm = LLM(**llm_kwargs)

    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=16),
    )

    scenarios = cfg["scenarios"]
    all_results = []
    temperature = cfg.get("temperature", 0.0)

    for scenario in scenarios:
        modality = scenario.get("modality", "text")

        if modality == "text":
            prompt_token_ids = scenario["prompt_token_ids"]
            output_lens = scenario["output_lens"]
            sp_list = [
                SamplingParams(temperature=temperature,
                               ignore_eos=True, max_tokens=ol)
                for ol in output_lens
            ]
            vllm_prompts = [dict(prompt_token_ids=p) for p in prompt_token_ids]
            start = time.perf_counter()
            outputs = llm.generate(vllm_prompts, sp_list)
            elapsed = time.perf_counter() - start
        else:
            mm_data = _preload_mm_data(
                scenario["dataset"], scenario["dataset_split"],
                scenario["num_seqs"], cfg["seed"],
            )
            max_input_tokens = cfg["max_model_len"] - scenario["output_len"]
            mm_data = _filter_and_prepare(
                mm_data, processor, max_input_tokens)
            print(f"  vLLM: {len(mm_data)} items after token-count filter "
                  f"(max_input={max_input_tokens})")
            sp_list = []
            vllm_prompts = []
            for item in mm_data:
                sp_list.append(
                    SamplingParams(temperature=temperature,
                                   ignore_eos=True,
                                   max_tokens=scenario["output_len"]))
                mm_dict = {}
                if item["images"] is not None:
                    mm_dict["image"] = item["images"]
                if item["video_frames"] is not None:
                    mm_dict["video"] = [
                        (item["video_frames"], item["video_metadata"])
                    ]
                vllm_prompts.append(dict(
                    prompt=item["chat_text"],
                    multi_modal_data=mm_dict,
                ))

            start = time.perf_counter()
            outputs = llm.generate(vllm_prompts, sp_list, use_tqdm=True)
            elapsed = time.perf_counter() - start

        total_prompt_tokens = sum(
            len(o.prompt_token_ids) if o.prompt_token_ids else 0
            for o in outputs
        )
        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs if c)
            for o in outputs
        )
        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {"text": o.outputs[0].text,
                 "token_ids": list(o.outputs[0].token_ids)}
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        modality = ls.get("modality", "text")

        if modality == "text":
            prompts = [dict(prompt_token_ids=p) for p in ls["prompt_token_ids"]]
            sp = SamplingParams(temperature=0.0,
                                ignore_eos=True, max_tokens=ls["output_len"])
            run_fn = lambda: llm.generate(prompts, sp, use_tqdm=False)
        else:
            mm_data = _preload_mm_data(
                ls["dataset"], ls["dataset_split"], 1, cfg["seed"],
            )
            max_lat_tokens = cfg["max_model_len"] - ls["output_len"]
            mm_data = _filter_and_prepare(
                mm_data, processor, max_lat_tokens)
            item = mm_data[0]
            sp = SamplingParams(temperature=0.0, ignore_eos=True,
                                max_tokens=ls["output_len"])
            mm_dict = {}
            if item["images"] is not None:
                mm_dict["image"] = item["images"]
            if item["video_frames"] is not None:
                mm_dict["video"] = [
                    (item["video_frames"], item["video_metadata"])
                ]
            lat_prompt = dict(prompt=item["chat_text"],
                              multi_modal_data=mm_dict)
            run_fn = lambda: llm.generate([lat_prompt], sp, use_tqdm=False)

        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            run_fn()
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            run_fn()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del llm

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Multi-scenario kb-nano subprocess worker (VLM, multi-modal)
# ---------------------------------------------------------------------------
KB_NANO_VLM_WORKER = _MM_PRELOAD_FN + r'''
import json, sys, time


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        cfg["model"], trust_remote_code=True)

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    engine = LlamaEngine(**engine_kwargs)

    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    import torch
    scenarios = cfg["scenarios"]
    all_results = []
    temperature = cfg.get("temperature", 0.0)
    top_p = cfg.get("top_p", 1.0)

    for scenario in scenarios:
        modality = scenario.get("modality", "text")

        if modality == "text":
            prompts = scenario["prompt_token_ids"]
            output_lens = scenario["output_lens"]
            sp_list = [
                SamplingParams(temperature=temperature, top_p=top_p,
                               max_tokens=ol, ignore_eos=True)
                for ol in output_lens
            ]
            engine.block_manager.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(prompts, sp_list, use_tqdm=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_input_tokens = sum(len(p) for p in prompts)
        else:
            mm_data = _preload_mm_data(
                scenario["dataset"], scenario["dataset_split"],
                scenario["num_seqs"], cfg["seed"],
            )
            max_input_tokens = cfg["max_model_len"] - scenario["output_len"]
            mm_data = _filter_and_prepare(
                mm_data, processor, max_input_tokens)
            print(f"  kb-nano: {len(mm_data)} items after token-count filter "
                  f"(max_input={max_input_tokens})")

            prompts = [item["prompt"] for item in mm_data]
            batch_images = []
            batch_videos = []
            for item in mm_data:
                if item["images"] is not None:
                    batch_images.append(item["images"])
                else:
                    batch_images.append(None)
                if item["video_frames"] is not None:
                    frames_pil = [
                        Image.fromarray(item["video_frames"][j]).convert("RGB")
                        for j in range(item["video_frames"].shape[0])
                    ]
                    batch_videos.append([frames_pil])
                else:
                    batch_videos.append(None)

            sp_list = [
                SamplingParams(temperature=temperature, top_p=top_p,
                               max_tokens=scenario["output_len"],
                               ignore_eos=True)
            ] * len(mm_data)

            total_input_tokens = 0
            engine.block_manager.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(prompts, sp_list,
                                      images=batch_images,
                                      videos=batch_videos,
                                      use_tqdm=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        total_output_tokens = sum(len(o.token_ids) for o in outputs)

        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_input_tokens": total_input_tokens if modality == "text" else 0,
            "total_output_tokens": total_output_tokens,
            "outputs": [
                {"generated_text": o.generated_text,
                 "token_ids": o.token_ids}
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        modality = ls.get("modality", "text")

        if modality == "text":
            prompts = ls["prompt_token_ids"]
            sp = SamplingParams(temperature=0.0, ignore_eos=True,
                                max_tokens=ls["output_len"])
            def run_fn():
                engine.block_manager.reset()
                torch.cuda.synchronize()
                engine.generate(prompts, sp)
                torch.cuda.synchronize()
        else:
            mm_data = _preload_mm_data(
                ls["dataset"], ls["dataset_split"], 1, cfg["seed"],
            )
            max_lat_tokens = cfg["max_model_len"] - ls["output_len"]
            mm_data = _filter_and_prepare(
                mm_data, processor, max_lat_tokens)
            item = mm_data[0]
            sp = SamplingParams(temperature=0.0, ignore_eos=True,
                                max_tokens=ls["output_len"])
            lat_images = None
            lat_videos = None
            if item["images"] is not None:
                lat_images = [item["images"]]
            if item["video_frames"] is not None:
                lat_frames_pil = [
                    Image.fromarray(item["video_frames"][j]).convert("RGB")
                    for j in range(item["video_frames"].shape[0])
                ]
                lat_videos = [[lat_frames_pil]]
            def run_fn(p=item["prompt"], imgs=lat_images, vids=lat_videos):
                engine.block_manager.reset()
                torch.cuda.synchronize()
                engine.generate(
                    [p], sp,
                    images=imgs,
                    videos=vids,
                )
                torch.cuda.synchronize()

        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            run_fn()
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            run_fn()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "output_len": ls["output_len"],
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

    del engine

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Multi-scenario vLLM subprocess worker (Whisper, audio)
# ---------------------------------------------------------------------------
VLLM_WHISPER_WORKER = r'''
import json, os, sys, time
import numpy as np
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

def _load_librispeech(dataset_name, dataset_split, num_seqs, seed):
    """Load audio samples from LibriSpeech and return as list of numpy arrays."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
    ds = ds.shuffle(seed=seed)
    samples = []
    for item in ds:
        audio = item["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        samples.append({"audio": arr, "sampling_rate": sr, "text": item["text"]})
        if len(samples) >= num_seqs:
            break
    return samples

def main():
    from vllm import LLM, SamplingParams

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    llm_kwargs = dict(
        model=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
    )
    if cfg.get("load_format"):
        llm_kwargs["load_format"] = cfg["load_format"]
    llm = LLM(**llm_kwargs)

    from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt

    # Warmup
    dummy_audio = np.zeros(16000, dtype=np.float32)
    warmup_prompt = ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt="",
            multi_modal_data={"audio": (dummy_audio, 16000)},
        ),
        decoder_prompt=TextPrompt(
            prompt="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        ),
    )
    llm.generate(
        [warmup_prompt],
        SamplingParams(temperature=0.0, max_tokens=16),
    )

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        num_seqs = scenario["num_seqs"]
        output_len = scenario["output_len"]

        audio_samples = _load_librispeech(
            scenario["dataset"], scenario["dataset_split"],
            num_seqs, cfg["seed"],
        )
        print(f"  Loaded {len(audio_samples)} audio samples from "
              f"{scenario['dataset']} ({scenario['dataset_split']})")

        prompts = []
        total_audio_s = 0.0
        for sample in audio_samples:
            audio, sr = sample["audio"], sample["sampling_rate"]
            total_audio_s += len(audio) / sr
            prompt = ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    prompt="",
                    multi_modal_data={"audio": (audio, sr)},
                ),
                decoder_prompt=TextPrompt(
                    prompt="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                ),
            )
            prompts.append(prompt)

        sp = SamplingParams(
            temperature=0.0, ignore_eos=True, max_tokens=output_len,
        )

        start = time.perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        elapsed = time.perf_counter() - start

        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs if c)
            for o in outputs
        )
        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_output_tokens": total_output_tokens,
            "num_seqs": len(audio_samples),
            "total_audio_duration_s": total_audio_s,
            "outputs": [
                {"text": o.outputs[0].text,
                 "token_ids": list(o.outputs[0].token_ids)}
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        output_len = ls["output_len"]
        batch_size = ls.get("batch_size", 1)
        audio_samples = _load_librispeech(
            ls["dataset"], ls["dataset_split"], batch_size, cfg["seed"] + 200,
        )
        prompts = []
        total_audio_s = 0.0
        for sample in audio_samples:
            audio, sr = sample["audio"], sample["sampling_rate"]
            total_audio_s += len(audio) / sr
            prompts.append(ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    prompt="",
                    multi_modal_data={"audio": (audio, sr)},
                ),
                decoder_prompt=TextPrompt(
                    prompt="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                ),
            ))
        sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            llm.generate(prompts, sp, use_tqdm=False)
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            llm.generate(prompts, sp, use_tqdm=False)
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": batch_size,
            "audio_duration_s": round(total_audio_s, 2),
            "output_len": output_len,
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del llm
    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Multi-scenario kb-nano subprocess worker (Whisper, audio)
# ---------------------------------------------------------------------------
KB_NANO_WHISPER_WORKER = r'''
import json, sys, time
import numpy as np

def _load_librispeech(dataset_name, dataset_split, num_seqs, seed):
    """Load audio samples from LibriSpeech and return as list of numpy arrays."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
    ds = ds.shuffle(seed=seed)
    samples = []
    for item in ds:
        audio = item["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        samples.append({"audio": arr, "sampling_rate": sr, "text": item["text"]})
        if len(samples) >= num_seqs:
            break
    return samples

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", True),
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    engine = LlamaEngine(**engine_kwargs)

    import torch
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(cfg["model"])

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        num_seqs = scenario["num_seqs"]
        output_len = scenario["output_len"]

        audio_samples = _load_librispeech(
            scenario["dataset"], scenario["dataset_split"],
            num_seqs, cfg["seed"],
        )
        print(f"  Loaded {len(audio_samples)} audio samples from "
              f"{scenario['dataset']} ({scenario['dataset_split']})")

        audio_features_list = []
        total_audio_s = 0.0
        for sample in audio_samples:
            audio, sr = sample["audio"], sample["sampling_rate"]
            total_audio_s += len(audio) / sr
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            audio_features_list.append(inputs.input_features[0])

        decoder_prompt = processor.tokenizer.encode(
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            add_special_tokens=False,
        )
        decoder_prompts = [decoder_prompt] * len(audio_samples)

        sp = SamplingParams(
            temperature=0.0, ignore_eos=True, max_tokens=output_len,
        )

        engine.block_manager.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(
            decoder_prompts, sp,
            audio_features=audio_features_list, use_tqdm=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_output_tokens = sum(len(o.token_ids) for o in outputs)
        result = {
            "name": scenario["name"],
            "elapsed": elapsed,
            "total_output_tokens": total_output_tokens,
            "num_seqs": len(audio_samples),
            "total_audio_duration_s": total_audio_s,
            "outputs": [
                {"generated_text": o.generated_text,
                 "token_ids": o.token_ids}
                for o in outputs
            ],
        }
        all_results.append(result)

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        output_len = ls["output_len"]
        batch_size = ls.get("batch_size", 1)
        audio_samples = _load_librispeech(
            ls["dataset"], ls["dataset_split"], batch_size, cfg["seed"] + 200,
        )
        audio_feats = []
        total_audio_s = 0.0
        for sample in audio_samples:
            audio, sr = sample["audio"], sample["sampling_rate"]
            total_audio_s += len(audio) / sr
            inp = processor(audio, sampling_rate=sr, return_tensors="pt")
            audio_feats.append(inp.input_features[0])
        decoder_prompt = processor.tokenizer.encode(
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            add_special_tokens=False,
        )
        decoder_prompts = [decoder_prompt] * batch_size

        sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            engine.generate(
                decoder_prompts, sp, audio_features=audio_feats,
            )
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(
                decoder_prompts, sp, audio_features=audio_feats,
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": batch_size,
            "audio_duration_s": round(total_audio_s, 2),
            "output_len": output_len,
            "num_iters": num_iters,
            "latencies": latencies,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)

    del engine

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------
def compute_alignment(
    a_outputs: list[dict],
    b_outputs: list[dict],
) -> dict:
    """Compare per-request token_ids. Returns alignment statistics."""
    total_seqs = len(a_outputs)
    exact_matches = 0
    total_matching_tokens = 0
    total_output_tokens = 0

    for a, b in zip(a_outputs, b_outputs):
        a_ids = a["token_ids"]
        b_ids = b["token_ids"]
        out_len = max(len(a_ids), len(b_ids))
        total_output_tokens += out_len

        if a_ids == b_ids:
            exact_matches += 1
            total_matching_tokens += len(a_ids)
        else:
            min_len = min(len(a_ids), len(b_ids))
            matching = sum(1 for j in range(min_len) if a_ids[j] == b_ids[j])
            total_matching_tokens += matching

    avg_matching = total_matching_tokens / total_seqs if total_seqs else 0
    avg_output_len = total_output_tokens / total_seqs if total_seqs else 0

    return {
        "exact_matches": exact_matches,
        "total_seqs": total_seqs,
        "total_matching_tokens": total_matching_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_matching_tokens_per_request": avg_matching,
        "avg_output_len": avg_output_len,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Throughput & alignment benchmark: kb-nano baseline vs vLLM",
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic alignment)",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip the throughput phase (run latency only)")
    parser.add_argument("--skip-latency", action="store_true",
                        help="Skip the latency benchmark phase")
    parser.add_argument("--latency-iters", type=int, default=5,
                        help="Timed iterations per latency scenario (default: 5)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save per-scenario outputs and results JSON "
             "(default: tests/results/<gpu>/<model>_tp<tp>)",
    )
    parser.add_argument(
        "--modality", type=str, default="all",
        choices=["all", "text", "image", "video"],
        help="Run only scenarios matching this modality (VLM models only, default: all)",
    )
    args = parser.parse_args()

    if args.num_seqs is None:
        args.num_seqs = 100 if _is_whisper_model(args.model) else 1000

    gpu = _detect_gpu_name()
    is_vlm = _is_vlm_model(args.model)
    is_whisper = _is_whisper_model(args.model)

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu / f"{short}_tp{args.tp}")

    if is_whisper:
        throughput_scenarios = WHISPER_SCENARIOS
        latency_scenarios = WHISPER_LATENCY_SCENARIOS
    elif is_vlm:
        throughput_scenarios = VLM_SCENARIOS
        latency_scenarios = VLM_LATENCY_SCENARIOS
    else:
        throughput_scenarios = SCENARIOS
        latency_scenarios = LATENCY_SCENARIOS

    if is_vlm and not is_whisper and args.modality != "all":
        throughput_scenarios = [
            s for s in throughput_scenarios
            if s.get("modality", "text") == args.modality
        ]
        latency_scenarios = [
            s for s in latency_scenarios
            if s.get("modality", "text") == args.modality
        ]

    # Pre-generate all scenario data
    scenario_data = []
    global_max_seq_len = 0
    if not args.skip_throughput:
        for i, scenario in enumerate(throughput_scenarios):
            if is_whisper:
                output_len = scenario["output_len"]
                max_seq_len = output_len + 10  # decoder prompt + output
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                num_seqs = args.num_seqs
                if scenario.get("use_full_dataset"):
                    num_seqs = 999_999  # load all available samples
                scenario_data.append({
                    "name": scenario["name"],
                    "output_len": output_len,
                    "dataset": scenario["dataset"],
                    "dataset_split": scenario["dataset_split"],
                    "num_seqs": num_seqs,
                })
                continue

            modality = scenario.get("modality", "text") if is_vlm else "text"
            if modality == "text":
                rng_seed = args.seed + i
                random.seed(rng_seed)
                np.random.seed(rng_seed)
                input_len = scenario["input_len"]
                output_len = scenario["output_len"]
                prompt_token_ids = [
                    [randint(0, 10000) for _ in range(input_len)]
                    for _ in range(args.num_seqs)
                ]
                output_lens = [output_len] * args.num_seqs
                max_seq_len = input_len + output_len
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                scenario_data.append({
                    "name": scenario["name"],
                    "modality": "text",
                    "input_len": input_len,
                    "output_len": output_len,
                    "prompt_token_ids": prompt_token_ids,
                    "output_lens": output_lens,
                })
            else:
                # Image/video: dataset is loaded inside the subprocess worker.
                # Large images can produce many vision tokens; be generous.
                max_seq_len = 16384 + scenario["output_len"]
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                scenario_data.append({
                    "name": scenario["name"],
                    "modality": modality,
                    "output_len": scenario["output_len"],
                    "dataset": scenario["dataset"],
                    "dataset_split": scenario["dataset_split"],
                    "num_seqs": args.num_seqs,
                })

    # Pre-generate latency scenario data
    latency_data = []
    if not args.skip_latency:
        for j, ls in enumerate(latency_scenarios):
            if is_whisper:
                max_seq_len = ls["output_len"] + 10
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                latency_data.append({
                    "name": ls["name"],
                    "output_len": ls["output_len"],
                    "batch_size": ls["batch_size"],
                    "dataset": ls["dataset"],
                    "dataset_split": ls["dataset_split"],
                    "num_warmup": 3,
                    "num_iters": args.latency_iters,
                })
                continue

            modality = ls.get("modality", "text") if is_vlm else "text"
            if modality == "text":
                rng_seed = args.seed + 100 + j
                random.seed(rng_seed)
                np.random.seed(rng_seed)
                bs = ls["batch_size"]
                prompt_token_ids = [
                    [randint(0, 10000) for _ in range(ls["input_len"])]
                    for _ in range(bs)
                ]
                seq_len = ls["input_len"] + ls["output_len"]
                if seq_len > global_max_seq_len:
                    global_max_seq_len = seq_len
                latency_data.append({
                    "name": ls["name"],
                    "modality": "text",
                    "input_len": ls["input_len"],
                    "output_len": ls["output_len"],
                    "batch_size": bs,
                    "prompt_token_ids": prompt_token_ids,
                    "num_warmup": 3,
                    "num_iters": args.latency_iters,
                })
            else:
                max_seq_len = 16384 + ls["output_len"]
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len
                latency_data.append({
                    "name": ls["name"],
                    "modality": modality,
                    "output_len": ls["output_len"],
                    "batch_size": ls["batch_size"],
                    "dataset": ls["dataset"],
                    "dataset_split": ls["dataset_split"],
                    "num_warmup": 3,
                    "num_iters": args.latency_iters,
                })

    print("=" * 70)
    print("  kb-nano Baseline vs vLLM -- Multi-Scenario Benchmark")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    model_type_str = "Whisper" if is_whisper else ("VLM" if is_vlm else "LLM")
    print(f"  Model type     : {model_type_str}")
    if is_vlm and args.modality != "all":
        print(f"  Modality       : {args.modality}")
    print(f"  TP             : {args.tp}")
    has_full = any(s.get("use_full_dataset") for s in throughput_scenarios) if is_whisper else False
    seqs_label = "full dataset" if has_full else str(args.num_seqs)
    print(f"  Seqs/scenario  : {seqs_label}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Enforce eager  : {args.enforce_eager}")
    print(f"  Seed           : {args.seed}")
    print(f"  Max seq len    : {global_max_seq_len}")
    print(f"  Output dir     : {args.output_dir}")
    if not args.skip_throughput:
        print(f"  Scenarios      : {', '.join(s['name'] for s in throughput_scenarios)}")
    else:
        print(f"  Scenarios      : (throughput skipped)")
    if latency_data:
        print(f"  Latency        : {', '.join(s['name'] for s in latency_scenarios)}"
              f" ({args.latency_iters} iters)")
    print("=" * 70)

    if is_whisper:
        vllm_worker = VLLM_WHISPER_WORKER
        kb_worker = KB_NANO_WHISPER_WORKER
    elif is_vlm:
        vllm_worker = VLLM_VLM_WORKER
        kb_worker = KB_NANO_VLM_WORKER
    else:
        vllm_worker = VLLM_WORKER
        kb_worker = KB_NANO_WORKER

    if is_whisper:
        global_max_seq_len = 448  # Whisper max_target_positions

    # -- Run vLLM (one subprocess, all scenarios) --
    vllm_raw = None
    if not args.skip_vllm:
        short_name = args.model.split("/")[-1]
        vllm_config = {
            "model": args.model,
            "tp": args.tp,
            "seed": args.seed,
            "temperature": args.temperature,
            "enforce_eager": args.enforce_eager,
            "max_model_len": global_max_seq_len,
            "scenarios": scenario_data,
            "latency_scenarios": latency_data,
            # "load_format": "fastsafetensors",  # disabled: OOM for large models
        }
        vllm_raw = run_worker(
            vllm_worker, vllm_config,
            f"vLLM [{short_name}] all scenarios (TP={args.tp})",
        )

    # -- Run kb-nano (one subprocess, all scenarios) --
    kb_root = str(_PROJECT_ROOT)
    package_name = _PACKAGE_DIR.name
    kb_config = {
        "model": args.model,
        "tp": args.tp,
        "seed": args.seed,
        "temperature": args.temperature,
        "enforce_eager": args.enforce_eager,
        "max_model_len": global_max_seq_len,
        "project_root": kb_root,
        "package_name": package_name,
        "scenarios": scenario_data,
        "latency_scenarios": latency_data,
    }
    short_name = args.model.split("/")[-1]
    kb_raw = run_worker(
        kb_worker, kb_config,
        f"kb-nano [{short_name}] all scenarios (TP={args.tp})",
    )
    if kb_raw is None:
        print("  ERROR: kb-nano subprocess failed.")
        sys.exit(1)

    kb_latency = kb_raw.get("latency", [])
    vllm_latency = vllm_raw.get("latency", []) if vllm_raw else []

    # -- Compute throughput metrics per scenario --
    all_results = []
    if not args.skip_throughput:
        kb_results = kb_raw["throughput"]
        vllm_results = vllm_raw["throughput"] if vllm_raw else None

        for i, scenario in enumerate(throughput_scenarios):
            kb_data = kb_results[i]
            kb_tps = kb_data["total_output_tokens"] / kb_data["elapsed"]

            result = {
                "scenario": scenario["name"],
                "num_seqs": kb_data.get("num_seqs", args.num_seqs),
                "kb_nano_elapsed": kb_data["elapsed"],
                "kb_nano_output_tokens": kb_data["total_output_tokens"],
                "kb_nano_tok_per_s": kb_tps,
            }
            if "input_len" in scenario:
                result["input_len"] = scenario["input_len"]
            if is_whisper:
                result["total_audio_duration_s"] = kb_data.get(
                    "total_audio_duration_s", 0)
            result["output_len"] = scenario["output_len"]

            if vllm_results is not None:
                v_data = vllm_results[i]
                v_tps = v_data["total_output_tokens"] / v_data["elapsed"]
                speedup = kb_tps / v_tps
                result["vllm_elapsed"] = v_data["elapsed"]
                result["vllm_output_tokens"] = v_data["total_output_tokens"]
                result["vllm_tok_per_s"] = v_tps
                result["speedup"] = speedup

                if args.temperature == 0.0:
                    alignment = compute_alignment(
                        kb_data["outputs"], v_data["outputs"]
                    )
                    result["alignment"] = alignment

            if args.output_dir:
                scenario_dir = os.path.join(args.output_dir, scenario["name"])
                os.makedirs(scenario_dir, exist_ok=True)

                kb_out_path = os.path.join(scenario_dir, "kb_nano_outputs.json")
                with open(kb_out_path, "w") as f:
                    json.dump(kb_data, f, indent=2)

                if vllm_results is not None:
                    vllm_out_path = os.path.join(scenario_dir, "vllm_outputs.json")
                    with open(vllm_out_path, "w") as f:
                        json.dump(vllm_results[i], f, indent=2)

            all_results.append(result)

        print(f"\n\n{'=' * 90}")
        print("  THROUGHPUT SUMMARY")
        print(f"{'=' * 90}")
        if is_whisper:
            header = (
                f"  {'SCENARIO':<16} {'SEQS':>5} {'AUDIO':>8} {'OUT':>5} "
                f"{'KB-NANO tok/s':>15} {'vLLM tok/s':>12} {'SPEEDUP':>8} "
                f"{'AVG MATCH TOKS':>15}"
            )
        elif is_vlm:
            header = (
                f"  {'SCENARIO':<16} {'OUT':>5} "
                f"{'KB-NANO tok/s':>15} {'vLLM tok/s':>12} {'SPEEDUP':>8} "
                f"{'AVG MATCH TOKS':>15}"
            )
        else:
            header = (
                f"  {'SCENARIO':<16} {'IN':>5} {'OUT':>5} "
                f"{'KB-NANO tok/s':>15} {'vLLM tok/s':>12} {'SPEEDUP':>8} "
                f"{'AVG MATCH TOKS':>15}"
            )
        print(header)
        print(f"  {'-' * 84}")

        for r in all_results:
            kb_tps_str = f"{r['kb_nano_tok_per_s']:,.0f}"
            v_tps_str = (
                f"{r['vllm_tok_per_s']:,.0f}" if "vllm_tok_per_s" in r else "N/A"
            )
            speedup_str = f"{r['speedup']:.2f}x" if "speedup" in r else "N/A"

            align = r.get("alignment", {})
            avg_match = align.get("avg_matching_tokens_per_request", 0)
            avg_out = align.get("avg_output_len", 0)
            if avg_out > 0:
                match_str = f"{avg_match:.1f}/{avg_out:.0f}"
            else:
                match_str = "N/A"

            if is_whisper:
                total_audio_s = r.get("total_audio_duration_s", 0)
                audio_min = total_audio_s / 60.0
                audio_str = f"{audio_min:.1f}m"
                print(
                    f"  {r['scenario']:<16} {r['num_seqs']:>5} {audio_str:>8} "
                    f"{r['output_len']:>5} "
                    f"{kb_tps_str:>15} {v_tps_str:>12} {speedup_str:>8} "
                    f"{match_str:>15}"
                )
            elif is_vlm:
                print(
                    f"  {r['scenario']:<16} {r['output_len']:>5} "
                    f"{kb_tps_str:>15} {v_tps_str:>12} {speedup_str:>8} "
                    f"{match_str:>15}"
                )
            else:
                print(
                    f"  {r['scenario']:<16} {r['input_len']:>5} {r['output_len']:>5} "
                    f"{kb_tps_str:>15} {v_tps_str:>12} {speedup_str:>8} "
                    f"{match_str:>15}"
                )

        print(f"{'=' * 90}")

    # -- Latency summary table --
    latency_combined = []
    if kb_latency:
        print(f"\n{'=' * 110}")
        print("  LATENCY SUMMARY")
        print(f"{'=' * 110}")
        print(
            f"  {'SCENARIO':<18} {'BS':>4} {'OUT':>5} {'ITERS':>6}"
            f"  {'KB-NANO med':>12} {'vLLM med':>12}"
            f"  {'KB-NANO ms/tok':>15} {'vLLM ms/tok':>12} {'SPEEDUP':>8}"
        )
        print(f"  {'-' * 100}")

        for i, kb_lat in enumerate(kb_latency):
            kb_lats = np.array(kb_lat["latencies"])
            kb_med = float(np.median(kb_lats))
            kb_p99 = float(np.percentile(kb_lats, 99))
            bs = kb_lat["batch_size"]
            out_len = kb_lat["output_len"]
            total_out_tokens = bs * out_len
            kb_ms_per_tok = (kb_med / total_out_tokens) * 1000

            lat_result = {
                "scenario": kb_lat["name"],
                "batch_size": bs,
                "output_len": out_len,
                "num_iters": kb_lat["num_iters"],
                "kb_nano_median_s": kb_med,
                "kb_nano_p99_s": kb_p99,
                "kb_nano_ms_per_tok": kb_ms_per_tok,
                "kb_nano_latencies": kb_lat["latencies"],
            }
            if "input_len" in kb_lat:
                lat_result["input_len"] = kb_lat["input_len"]

            v_med_str = "N/A"
            speedup_str = "N/A"
            v_ms_str = "N/A"
            if i < len(vllm_latency):
                v_lat = vllm_latency[i]
                v_lats = np.array(v_lat["latencies"])
                v_med = float(np.median(v_lats))
                v_p99 = float(np.percentile(v_lats, 99))
                v_ms_per_tok = (v_med / total_out_tokens) * 1000
                speedup = v_med / kb_med
                v_med_str = f"{v_med:.4f}s"
                speedup_str = f"{speedup:.2f}x"
                v_ms_str = f"{v_ms_per_tok:.2f}"
                lat_result["vllm_median_s"] = v_med
                lat_result["vllm_p99_s"] = v_p99
                lat_result["vllm_ms_per_tok"] = v_ms_per_tok
                lat_result["speedup"] = speedup
                lat_result["vllm_latencies"] = v_lat["latencies"]

            print(
                f"  {kb_lat['name']:<18} {bs:>4}"
                f" {out_len:>5} {kb_lat['num_iters']:>6}"
                f"  {kb_med:.4f}s{'':<3} {v_med_str:>12}"
                f"  {kb_ms_per_tok:>13.2f}   {v_ms_str:>10} {speedup_str:>8}"
            )
            latency_combined.append(lat_result)

        print(f"{'=' * 110}")

    # -- Save combined results --
    if args.output_dir and (all_results or latency_combined):
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "results.json")
        combined = {
            "gpu": gpu,
            "model": args.model,
            "model_type": "vlm" if is_vlm else "llm",
            "tp": args.tp,
            "seed": args.seed,
            "temperature": args.temperature,
            "num_seqs": args.num_seqs,
            "enforce_eager": args.enforce_eager,
        }
        if all_results:
            combined["scenarios"] = all_results
        if latency_combined:
            combined["latency_scenarios"] = latency_combined
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
