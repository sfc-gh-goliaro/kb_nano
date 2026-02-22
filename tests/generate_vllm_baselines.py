#!/usr/bin/env python3
"""Generate vLLM baselines for all models (text-only and multimodal).

Run once, save results to JSON. Then only rerun our engine during iteration.

Usage:
    # Text-only baselines
    python tests/generate_vllm_baselines.py --model Qwen/Qwen2-VL-7B-Instruct
    python tests/generate_vllm_baselines.py --model meta-llama/Llama-3.1-8B-Instruct

    # Multimodal baselines (images + video)
    python tests/generate_vllm_baselines.py --model Qwen/Qwen2-VL-7B-Instruct --mode mm
    python tests/generate_vllm_baselines.py --model Qwen/Qwen3-VL-8B-Instruct --mode mm
"""

import argparse
import json
import os
import time

import numpy as np

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

TEXT_PROMPTS = [
    "What is 2 + 2?",
    "Explain quantum entanglement to a 10-year-old.",
    "Write a Python async web scraper with error handling.",
    "Translate 'hello' into French, German, and Japanese.",
]

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def _chat(text):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _load_image_assets():
    from PIL import Image
    assets_dir = os.path.expanduser("~/.cache/vllm/assets/vllm_public_assets")
    stop_sign = Image.open(os.path.join(assets_dir, "stop_sign.jpg"))
    cherry_blossom = Image.open(os.path.join(assets_dir, "cherry_blossom.jpg"))
    return stop_sign, cherry_blossom


def _make_synthetic_image(width, height, color=(255, 0, 0)):
    from PIL import Image
    return Image.new("RGB", (width, height), color)


def _load_video_asset(num_frames=4):
    from vllm.assets.video import VideoAsset
    v = VideoAsset("baby_reading", num_frames=num_frames)
    return v


def _build_mm_test_cases(model_name):
    """Build multimodal test cases: list of (name, prompt_text, mm_data_dict)."""
    stop_sign, cherry_blossom = _load_image_assets()
    synth_256 = _make_synthetic_image(256, 256, (0, 0, 255))
    video_asset = _load_video_asset(num_frames=4)
    video_frames = video_asset.np_ndarrays

    is_qwen3 = "Qwen3" in model_name
    if is_qwen3:
        video_data = (video_frames, video_asset.metadata)
    else:
        video_data = video_frames

    cases = [
        (
            "image_stop_sign_short",
            _chat(f"{IMAGE_PLACEHOLDER}What is shown in this image?"),
            {"image": stop_sign},
        ),
        (
            "image_cherry_blossom_detail",
            _chat(f"{IMAGE_PLACEHOLDER}Describe this image in detail."),
            {"image": cherry_blossom},
        ),
        (
            "image_synth_256_short",
            _chat(f"{IMAGE_PLACEHOLDER}What color is this image?"),
            {"image": synth_256},
        ),
        (
            "multi_image_compare",
            _chat(
                f"{IMAGE_PLACEHOLDER}{IMAGE_PLACEHOLDER}"
                "Describe these two images separately. "
                "For each image, reply with a short sentence "
                "(no more than 10 words)."
            ),
            {"image": [stop_sign, cherry_blossom]},
        ),
        (
            "video_baby_reading_short",
            _chat(f"{VIDEO_PLACEHOLDER}Describe this video briefly."),
            {"video": video_data},
        ),
        (
            "video_baby_reading_detail",
            _chat(
                f"{VIDEO_PLACEHOLDER}"
                "What is happening in this video? Describe step by step."
            ),
            {"video": video_data},
        ),
    ]
    return cases


BATCH_PROMPTS = [
    "What is 2 + 2?",
    "Explain quantum entanglement to a 10-year-old.",
    "Write a Python async web scraper with error handling.",
    "Translate 'hello' into French, German, and Japanese.",
    "What are the three laws of thermodynamics?",
    "Write a haiku about machine learning.",
    "Explain the difference between TCP and UDP.",
    "What is the capital of Australia?",
    "List the planets in our solar system in order.",
    "Write a SQL query to find duplicate emails.",
    "What causes rainbows?",
    "Explain how a blockchain works.",
    "What is the Pythagorean theorem?",
    "Write a recursive function to compute Fibonacci numbers.",
    "What is the speed of light in meters per second?",
    "Describe the process of photosynthesis.",
]


def run_text_only(model_name, max_tokens=100, seed=42, tp=1,
                  enforce_eager=True, batched=False):
    from vllm import LLM, SamplingParams

    prompts = BATCH_PROMPTS if batched else TEXT_PROMPTS

    print(f"Loading vLLM with {model_name} (eager={enforce_eager}, batched={batched})...")
    llm = LLM(
        model=model_name, seed=seed, enforce_eager=enforce_eager,
        tensor_parallel_size=tp, trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)

    llm.generate(["warmup"], sp)

    if batched:
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        elapsed = time.perf_counter() - t0
        results = []
        for prompt, out in zip(prompts, outputs):
            results.append({
                "prompt": prompt,
                "text": out.outputs[0].text,
                "token_ids": list(out.outputs[0].token_ids),
            })
    else:
        results = []
        t0 = time.perf_counter()
        for prompt in prompts:
            out = llm.generate([prompt], sp)[0]
            results.append({
                "prompt": prompt,
                "text": out.outputs[0].text,
                "token_ids": list(out.outputs[0].token_ids),
            })
        elapsed = time.perf_counter() - t0

    total_tokens = sum(len(r["token_ids"]) for r in results)
    perf = {
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(total_tokens / elapsed, 1),
    }

    return {"prompts": prompts, "results": results, "perf": perf}


def run_multimodal(model_name, max_tokens=128, seed=42, tp=1,
                   enforce_eager=True):
    from vllm import LLM, SamplingParams

    print(f"Loading vLLM with {model_name} for multimodal baselines (eager={enforce_eager})...")
    llm = LLM(
        model=model_name, seed=seed, enforce_eager=enforce_eager,
        tensor_parallel_size=tp, trust_remote_code=True,
        limit_mm_per_prompt={"image": 4, "video": 2},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)

    llm.generate(["warmup"], sp)

    cases = _build_mm_test_cases(model_name)

    results = []
    t0 = time.perf_counter()
    for name, prompt_text, mm_data in cases:
        print(f"  Running case: {name}...")
        prompt = {"prompt": prompt_text, "multi_modal_data": mm_data}
        out = llm.generate([prompt], sp)[0]
        results.append({
            "name": name,
            "prompt_text": prompt_text,
            "text": out.outputs[0].text,
            "token_ids": list(out.outputs[0].token_ids),
        })
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(r["token_ids"]) for r in results)
    perf = {
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(total_tokens / elapsed, 1),
    }

    return {"results": results, "perf": perf}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["text", "mm", "batch"], default="text")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Use CUDA graphs (enforce_eager=False)")
    args = parser.parse_args()

    enforce_eager = not args.cuda_graphs

    baselines_dir = os.path.join(os.path.dirname(__file__), "baselines")
    os.makedirs(baselines_dir, exist_ok=True)
    short_name = args.model.split("/")[-1]

    suffix = "-cg" if args.cuda_graphs else ""

    if args.mode == "text":
        max_tokens = args.max_tokens or 100
        data = run_text_only(args.model, max_tokens, args.seed, args.tp,
                             enforce_eager=enforce_eager)
        out_path = os.path.join(baselines_dir, f"{short_name}{suffix}.json")
    elif args.mode == "batch":
        max_tokens = args.max_tokens or 100
        data = run_text_only(args.model, max_tokens, args.seed, args.tp,
                             enforce_eager=enforce_eager, batched=True)
        out_path = os.path.join(baselines_dir, f"{short_name}-batch{suffix}.json")
    else:
        max_tokens = args.max_tokens or 128
        data = run_multimodal(args.model, max_tokens, args.seed, args.tp,
                              enforce_eager=enforce_eager)
        out_path = os.path.join(baselines_dir, f"{short_name}-mm{suffix}.json")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nBaseline saved to {out_path}")
    print(f"Performance: {data['perf']}")
    for r in data["results"]:
        name = r.get("name", r.get("prompt", "")[:40])
        print(f"\n  Case: {name}")
        print(f"  Output: {r['text'][:80]}...")
        print(f"  Tokens: {len(r['token_ids'])}")


if __name__ == "__main__":
    main()
