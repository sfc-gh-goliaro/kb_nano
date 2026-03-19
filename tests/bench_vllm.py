#!/usr/bin/env python3
"""
Throughput and alignment benchmark: kb-nano baseline vs vLLM.

For LLM models: runs three text-only scenarios using real datasets:
  - LongBench (prefill-heavy, 500 reqs)
  - ShareGPT (short QA, 3000 reqs)
  - DS-1000 (long-output code gen, 1000 reqs)

For VLM models (Qwen2-VL, Qwen3-VL): runs three throughput scenarios
(text-only mixed from LLM datasets, image, video) and two latency
scenarios (single-image, single-video) using real multimodal datasets
(VisionArena, MMVU).

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
    {"name": "longbench-summ", "dataset": "longbench", "output_len": 512,
     "num_requests": 500},
    {"name": "sharegpt-short", "dataset": "sharegpt", "output_len": None,
     "num_requests": 3000},
    {"name": "ds1000-code",   "dataset": "ds1000",   "output_len": 8192,
     "num_requests": 1000},
]

LATENCY_SCENARIOS = [
    {"name": "single-short",       "dataset": "sharegpt",  "output_len": 128, "batch_size": 1},
    {"name": "single-long-context", "dataset": "longbench", "output_len": 128, "batch_size": 1},
]

VLM_SCENARIOS = [
    {"name": "text-only",  "modality": "text",  "dataset": "mixed", "output_len": None},
    {"name": "image",      "modality": "image", "output_len": 512,
     "dataset": "lmarena-ai/VisionArena-Chat", "dataset_split": "train"},
    {"name": "video",      "modality": "video", "output_len": 512,
     "dataset": "yale-nlp/MMVU", "dataset_split": "validation"},
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


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _load_longbench(tokenizer, num_requests: int, seed: int,
                    min_prompt_tokens: int = 12000) -> dict:
    """Load LongBench gov_report + multi_news, filter for long inputs."""
    from datasets import load_dataset

    items = []
    for subset in ("gov_report", "multi_news"):
        ds = load_dataset("THUDM/LongBench", subset, split="test",
                          trust_remote_code=True)
        for row in ds:
            text = row.get("context", "") or row.get("input", "")
            if not text:
                continue
            items.append(text)

    rng = random.Random(seed)
    rng.shuffle(items)

    prompts = []
    prompt_lens = []
    for text in items:
        if len(prompts) >= num_requests:
            break
        ids = tokenizer.encode(text)
        if len(ids) >= min_prompt_tokens:
            prompts.append(text)
            prompt_lens.append(len(ids))

    if len(prompts) < num_requests:
        print(f"  WARNING: LongBench only yielded {len(prompts)} prompts "
              f"with >= {min_prompt_tokens} tokens (requested {num_requests})")

    return {"prompts": prompts, "prompt_lens": prompt_lens}


def _load_sharegpt(tokenizer, num_requests: int, seed: int) -> dict:
    """Load ShareGPT single-turn conversations."""
    from datasets import load_dataset

    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split="train",
        trust_remote_code=True,
    )

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    prompts = []
    prompt_lens = []
    output_lens = []
    for idx in indices:
        if len(prompts) >= num_requests:
            break
        row = ds[idx]
        convs = row.get("conversations", [])
        if len(convs) < 2:
            continue
        prompt = convs[0].get("value", "")
        completion = convs[1].get("value", "")
        if not prompt or not completion:
            continue
        # Filter to reasonable lengths
        prompt_ids = tokenizer.encode(prompt)
        completion_ids = tokenizer.encode(completion)
        if len(prompt_ids) < 4 or len(prompt_ids) > 2048:
            continue
        if len(completion_ids) < 4:
            continue
        prompts.append(prompt)
        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(completion_ids))

    return {"prompts": prompts, "prompt_lens": prompt_lens,
            "output_lens": output_lens}


def _load_ds1000(tokenizer, num_requests: int, seed: int) -> dict:
    """Load DS-1000 code generation prompts."""
    from datasets import load_dataset

    ds = load_dataset("xlangai/DS-1000", split="test",
                      trust_remote_code=True)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    prompts = []
    prompt_lens = []
    for idx in indices:
        if len(prompts) >= num_requests:
            break
        row = ds[idx]
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        prompt_ids = tokenizer.encode(prompt)
        if len(prompt_ids) < 4:
            continue
        prompts.append(prompt)
        prompt_lens.append(len(prompt_ids))

    return {"prompts": prompts, "prompt_lens": prompt_lens}


def _load_text_datasets(model: str, seed: int) -> dict:
    """Load all text datasets and return structured data.

    Returns dict with keys: "longbench", "sharegpt", "ds1000", each containing
    {"prompts": list[str], "prompt_lens": list[int], "output_lens": list[int] (sharegpt only)}.
    """
    from transformers import AutoTokenizer
    print("  Loading tokenizer and text datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    longbench = _load_longbench(tokenizer, num_requests=500, seed=seed)
    print(f"    LongBench: {len(longbench['prompts'])} prompts loaded")

    sharegpt = _load_sharegpt(tokenizer, num_requests=3000, seed=seed)
    print(f"    ShareGPT:  {len(sharegpt['prompts'])} prompts loaded")

    ds1000 = _load_ds1000(tokenizer, num_requests=1000, seed=seed)
    print(f"    DS-1000:   {len(ds1000['prompts'])} prompts loaded")

    return {
        "longbench": longbench,
        "sharegpt": sharegpt,
        "ds1000": ds1000,
    }


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
    llm = LLM(
        model=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
    )

    # Warmup
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=16),
    )

    scenarios = cfg["scenarios"]
    all_results = []
    for scenario in scenarios:
        prompts = scenario["prompts"]
        output_lens = scenario["output_lens"]
        temperature = cfg.get("temperature", 0.0)

        sp_list = [
            SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=ol)
            for ol in output_lens
        ]

        vllm_prompts = [dict(prompt=p) for p in prompts]
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
        prompts = [dict(prompt=p) for p in ls["prompts"]]
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
        prompts = scenario["prompts"]
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

        total_input_tokens = sum(
            len(engine.tokenizer.encode(p)) if isinstance(p, str) else len(p)
            for p in prompts
        )
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
        prompts = ls["prompts"]
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

    llm = LLM(
        model=model_name,
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
    )

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
            prompts = scenario["prompts"]
            output_lens = scenario["output_lens"]
            sp_list = [
                SamplingParams(temperature=temperature,
                               ignore_eos=True, max_tokens=ol)
                for ol in output_lens
            ]
            vllm_prompts = [dict(prompt=p) for p in prompts]
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
            prompts = [dict(prompt=p) for p in ls["prompts"]]
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
            prompts = scenario["prompts"]
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
            total_input_tokens = sum(
                len(engine.tokenizer.encode(p)) if isinstance(p, str) else len(p)
                for p in prompts
            )
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
            prompts = ls["prompts"]
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

    gpu = _detect_gpu_name()
    is_vlm = _is_vlm_model(args.model)

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu / f"{short}_tp{args.tp}")

    throughput_scenarios = VLM_SCENARIOS if is_vlm else SCENARIOS
    latency_scenarios = VLM_LATENCY_SCENARIOS if is_vlm else LATENCY_SCENARIOS

    if is_vlm and args.modality != "all":
        throughput_scenarios = [
            s for s in throughput_scenarios
            if s.get("modality", "text") == args.modality
        ]
        latency_scenarios = [
            s for s in latency_scenarios
            if s.get("modality", "text") == args.modality
        ]

    # Load real text datasets (shared across all text scenarios)
    has_text_throughput = not args.skip_throughput and any(
        s.get("modality", "text") == "text" if is_vlm else True
        for s in throughput_scenarios
    )
    has_text_latency = not args.skip_latency and any(
        s.get("modality", "text") == "text" if is_vlm else True
        for s in latency_scenarios
    )
    text_data = None
    if has_text_throughput or has_text_latency:
        text_data = _load_text_datasets(args.model, args.seed)

    # Pre-generate all scenario data
    scenario_data = []
    global_max_seq_len = 0
    if not args.skip_throughput:
        for i, scenario in enumerate(throughput_scenarios):
            modality = scenario.get("modality", "text") if is_vlm else "text"
            if modality == "text":
                ds_key = scenario.get("dataset", "mixed")
                num_reqs = scenario.get("num_requests", 1000)

                if ds_key == "mixed":
                    # VLM text-only: proportional mix from all three datasets
                    all_prompts = []
                    all_prompt_lens = []
                    all_output_lens = []
                    for src_key, src_scenario in zip(
                        ["longbench", "sharegpt", "ds1000"], SCENARIOS
                    ):
                        src = text_data[src_key]
                        src_output_len = src_scenario["output_len"]
                        src_num = min(
                            src_scenario.get("num_requests", 1000),
                            len(src["prompts"]),
                        )
                        # Proportionally scale to num_reqs
                        n = min(
                            int(src_num * num_reqs / 4500),
                            len(src["prompts"]),
                        )
                        all_prompts.extend(src["prompts"][:n])
                        all_prompt_lens.extend(src["prompt_lens"][:n])
                        if "output_lens" in src and src_output_len is None:
                            all_output_lens.extend(src["output_lens"][:n])
                        else:
                            ol = src_output_len if src_output_len is not None else 512
                            all_output_lens.extend([ol] * n)

                    prompts = all_prompts
                    prompt_lens = all_prompt_lens
                    output_lens = all_output_lens
                else:
                    src = text_data[ds_key]
                    n = min(num_reqs, len(src["prompts"]))
                    prompts = src["prompts"][:n]
                    prompt_lens = src["prompt_lens"][:n]

                    output_len = scenario["output_len"]
                    if output_len is not None:
                        output_lens = [output_len] * n
                    else:
                        # Dynamic from dataset (ShareGPT)
                        output_lens = src["output_lens"][:n]

                max_prompt_len = max(prompt_lens) if prompt_lens else 0
                max_output_len = max(output_lens) if output_lens else 0
                max_seq_len = max_prompt_len + max_output_len
                if max_seq_len > global_max_seq_len:
                    global_max_seq_len = max_seq_len

                scenario_data.append({
                    "name": scenario["name"],
                    "modality": "text",
                    "prompts": prompts,
                    "output_lens": output_lens,
                    "num_requests": len(prompts),
                    "avg_prompt_len": sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0,
                    "max_prompt_len": max_prompt_len,
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
                    "num_seqs": scenario.get("num_requests", 1000),
                })

    # Pre-generate latency scenario data
    latency_data = []
    if not args.skip_latency:
        for j, ls in enumerate(latency_scenarios):
            modality = ls.get("modality", "text") if is_vlm else "text"
            if modality == "text":
                ds_key = ls["dataset"]
                src = text_data[ds_key]
                bs = ls["batch_size"]
                # Pick prompt(s) from the end of the dataset to avoid overlap
                # with throughput data
                prompts = src["prompts"][-bs:]
                prompt_lens = src["prompt_lens"][-bs:]

                max_prompt_len = max(prompt_lens) if prompt_lens else 0
                seq_len = max_prompt_len + ls["output_len"]
                if seq_len > global_max_seq_len:
                    global_max_seq_len = seq_len
                latency_data.append({
                    "name": ls["name"],
                    "modality": "text",
                    "output_len": ls["output_len"],
                    "batch_size": bs,
                    "prompts": prompts,
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
    print(f"  Model type     : {'VLM' if is_vlm else 'LLM'}")
    if is_vlm and args.modality != "all":
        print(f"  Modality       : {args.modality}")
    print(f"  TP             : {args.tp}")
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

    vllm_worker = VLLM_VLM_WORKER if is_vlm else VLLM_WORKER
    kb_worker = KB_NANO_VLM_WORKER if is_vlm else KB_NANO_WORKER

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

            sd = scenario_data[i] if i < len(scenario_data) else {}
            num_reqs = sd.get("num_requests", 1000)
            avg_prompt_len = sd.get("avg_prompt_len", 0)

            result = {
                "scenario": scenario["name"],
                "num_requests": num_reqs,
                "kb_nano_elapsed": kb_data["elapsed"],
                "kb_nano_output_tokens": kb_data["total_output_tokens"],
                "kb_nano_tok_per_s": kb_tps,
                "avg_prompt_len": int(avg_prompt_len),
            }
            if "output_len" in scenario and scenario["output_len"] is not None:
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

        print(f"\n\n{'=' * 100}")
        print("  THROUGHPUT SUMMARY")
        print(f"{'=' * 100}")
        header = (
            f"  {'SCENARIO':<20} {'REQS':>5} {'AVG IN':>7} "
            f"{'KB-NANO tok/s':>15} {'vLLM tok/s':>12} {'SPEEDUP':>8} "
            f"{'AVG MATCH TOKS':>15}"
        )
        print(header)
        print(f"  {'-' * 94}")

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

            print(
                f"  {r['scenario']:<20} {r['num_requests']:>5} "
                f"{r['avg_prompt_len']:>7} "
                f"{kb_tps_str:>15} {v_tps_str:>12} {speedup_str:>8} "
                f"{match_str:>15}"
            )

        print(f"{'=' * 100}")

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
