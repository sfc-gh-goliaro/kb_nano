#!/usr/bin/env python3
"""Test our engine against cached vLLM baselines (text-only, multimodal, and batch).

Loads cached vLLM baselines from JSON, runs our engine, compares token-for-token.
Reports explicit PASS/FAIL for alignment (>= 15 tokens match) and performance (>= 1.0x).

Usage:
    # Text-only (sequential, eager)
    python tests/test_against_baselines.py --model Qwen/Qwen2-VL-7B-Instruct

    # Text-only with CUDA graphs
    python tests/test_against_baselines.py --model Qwen/Qwen2-VL-7B-Instruct --cuda-graphs

    # Batched (high-concurrency)
    python tests/test_against_baselines.py --model Qwen/Qwen2-VL-7B-Instruct --mode batch

    # Multimodal
    python tests/test_against_baselines.py --model Qwen/Qwen2-VL-7B-Instruct --mode mm

    # All modes
    python tests/test_against_baselines.py --model Qwen/Qwen2-VL-7B-Instruct --mode all
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

MIN_ALIGNED_TOKENS = 15


def _load_image_assets():
    from PIL import Image
    assets_dir = os.path.expanduser("~/.cache/vllm/assets/vllm_public_assets")
    stop_sign = Image.open(os.path.join(assets_dir, "stop_sign.jpg"))
    cherry_blossom = Image.open(os.path.join(assets_dir, "cherry_blossom.jpg"))
    return stop_sign, cherry_blossom


def _make_synthetic_image(width, height, color=(255, 0, 0)):
    from PIL import Image
    return Image.new("RGB", (width, height), color)


def _load_video_frames(num_frames=4):
    from vllm.assets.video import VideoAsset
    v = VideoAsset("baby_reading", num_frames=num_frames)
    return [frame for frame in v.pil_images]


def _build_mm_inputs():
    stop_sign, cherry_blossom = _load_image_assets()
    synth_256 = _make_synthetic_image(256, 256, (0, 0, 255))
    video_frames = _load_video_frames(num_frames=4)

    cases = [
        ("image_stop_sign_short",
         "What is shown in this image?",
         [stop_sign], None),
        ("image_cherry_blossom_detail",
         "Describe this image in detail.",
         [cherry_blossom], None),
        ("image_synth_256_short",
         "What color is this image?",
         [synth_256], None),
        ("multi_image_compare",
         "Describe these two images separately. "
         "For each image, reply with a short sentence "
         "(no more than 10 words).",
         [stop_sign, cherry_blossom], None),
        ("video_baby_reading_short",
         "Describe this video briefly.",
         None, [video_frames]),
        ("video_baby_reading_detail",
         "What is happening in this video? Describe step by step.",
         None, [video_frames]),
    ]
    return cases


def _check_alignment(v_ids, s_ids):
    """Return (divergence_point, is_pass) where is_pass = divergence >= MIN_ALIGNED_TOKENS."""
    min_len = min(len(v_ids), len(s_ids))
    div = next((j for j in range(min_len) if v_ids[j] != s_ids[j]), min_len)
    return div, div >= MIN_ALIGNED_TOKENS


def _print_results(label, results_pairs, baseline_perf, elapsed, total_tokens):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    align_pass = 0
    align_fail = 0
    for i, (name, br, our) in enumerate(results_pairs):
        v_ids = br["token_ids"]
        s_ids = our["token_ids"]
        div, ok = _check_alignment(v_ids, s_ids)

        if v_ids == s_ids:
            status = "EXACT MATCH"
            align_pass += 1
        elif ok:
            status = f"ALIGN OK (diverge at {div})"
            align_pass += 1
        else:
            status = f"ALIGN FAIL (diverge at {div} < {MIN_ALIGNED_TOKENS})"
            align_fail += 1

        print(f"  #{i} {status:35s} | {name}")
        if not ok:
            print(f"       vLLM : {br['text'][:70]!r}")
            print(f"       Ours : {our['text'][:70]!r}")

    our_tok_per_s = round(total_tokens / elapsed, 1) if elapsed > 0 else 0
    vllm_tps = baseline_perf["tok_per_s"]
    ratio = round(our_tok_per_s / vllm_tps, 2) if vllm_tps > 0 else 0
    perf_ok = ratio >= 1.0

    n = len(results_pairs)
    print(f"\n  Alignment: {align_pass}/{n} pass (>= {MIN_ALIGNED_TOKENS} tokens match)")
    print(f"  Performance: {our_tok_per_s} tok/s vs vLLM {vllm_tps} tok/s = {ratio}x"
          f"  {'PASS' if perf_ok else 'FAIL'}")
    print(f"{'=' * 70}")
    return align_fail, perf_ok


def run_text_only(engine, sp, baseline, short_name, batched=False):
    prompts = baseline["prompts"]

    if batched:
        t0 = time.perf_counter()
        outputs = engine.generate(prompts, sp)
        elapsed = time.perf_counter() - t0
        results = [{"text": o.generated_text, "token_ids": o.token_ids} for o in outputs]
    else:
        results = []
        t0 = time.perf_counter()
        for prompt in prompts:
            out = engine.generate([prompt], sp)[0]
            results.append({"text": out.generated_text, "token_ids": out.token_ids})
        elapsed = time.perf_counter() - t0

    total_tokens = sum(len(r["token_ids"]) for r in results)

    pairs = []
    for i, (br, our) in enumerate(zip(baseline["results"], results)):
        name = br.get("prompt", prompts[i])[:55]
        pairs.append((name, br, our))

    mode_label = "batch" if batched else "text"
    align_fail, perf_ok = _print_results(
        f"{short_name} ({mode_label})", pairs,
        baseline["perf"], elapsed, total_tokens)
    return align_fail, perf_ok


def run_multimodal(engine, sp, baseline, short_name):
    mm_inputs = _build_mm_inputs()
    baseline_results = baseline["results"]
    name_to_baseline = {r["name"]: r for r in baseline_results}

    image_pairs = []
    video_pairs = []
    results = []

    t0 = time.perf_counter()
    for name, prompt_text, images, videos in mm_inputs:
        out = engine.generate(
            [prompt_text], sp,
            images=[images] if images else None,
            videos=[videos] if videos else None,
        )[0]
        our = {"text": out.generated_text, "token_ids": out.token_ids}
        results.append(our)

        br = name_to_baseline.get(name)
        if br is None:
            print(f"  WARNING: no baseline for case '{name}', skipping comparison")
            continue

        pair = (name, br, our)
        if "video" in name:
            video_pairs.append(pair)
        else:
            image_pairs.append(pair)

    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(r["token_ids"]) for r in results)

    total_align_fail = 0
    perf_ok = True
    if image_pairs:
        af, pok = _print_results(
            f"{short_name} (images)", image_pairs,
            baseline["perf"], elapsed, total_tokens)
        total_align_fail += af
        perf_ok = perf_ok and pok
    if video_pairs:
        af, pok = _print_results(
            f"{short_name} (video)", video_pairs,
            baseline["perf"], elapsed, total_tokens)
        total_align_fail += af
        perf_ok = perf_ok and pok

    return total_align_fail, perf_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["text", "mm", "batch", "all"], default="text")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Test with CUDA graphs (enforce_eager=False)")
    args = parser.parse_args()

    enforce_eager = not args.cuda_graphs
    short_name = args.model.split("/")[-1]
    baselines_dir = os.path.join(os.path.dirname(__file__), "baselines")
    suffix = "-cg" if args.cuda_graphs else ""

    from kb_nano.engine import LlamaEngine, SamplingParams
    engine = LlamaEngine(
        model_name=args.model, seed=args.seed,
        enforce_eager=enforce_eager, tensor_parallel_size=args.tp,
    )

    engine.generate(["warmup"], SamplingParams(
        temperature=0.0, max_tokens=2, seed=args.seed))

    total_align_fail = 0
    all_perf_ok = True

    if args.mode in ("text", "all"):
        max_tokens = args.max_tokens or 100
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=args.seed)
        path = os.path.join(baselines_dir, f"{short_name}{suffix}.json")
        if os.path.exists(path):
            with open(path) as f:
                baseline = json.load(f)
            print(f"Loaded text baseline: {len(baseline['results'])} prompts")
            af, pok = run_text_only(engine, sp, baseline, short_name)
            total_align_fail += af
            all_perf_ok = all_perf_ok and pok
        else:
            print(f"No text baseline at {path}, skipping")

    if args.mode in ("batch", "all"):
        max_tokens = args.max_tokens or 100
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=args.seed)
        path = os.path.join(baselines_dir, f"{short_name}-batch{suffix}.json")
        if os.path.exists(path):
            with open(path) as f:
                baseline = json.load(f)
            print(f"Loaded batch baseline: {len(baseline['results'])} prompts")
            af, pok = run_text_only(engine, sp, baseline, short_name, batched=True)
            total_align_fail += af
            all_perf_ok = all_perf_ok and pok
        else:
            print(f"No batch baseline at {path}, skipping")

    if args.mode in ("mm", "all"):
        max_tokens = args.max_tokens or 128
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=args.seed)
        path = os.path.join(baselines_dir, f"{short_name}-mm{suffix}.json")
        if os.path.exists(path):
            with open(path) as f:
                baseline = json.load(f)
            print(f"Loaded multimodal baseline: {len(baseline['results'])} cases")
            af, pok = run_multimodal(engine, sp, baseline, short_name)
            total_align_fail += af
            all_perf_ok = all_perf_ok and pok
        else:
            print(f"No multimodal baseline at {path}, skipping")

    del engine

    print(f"\n{'#' * 70}")
    print(f"  OVERALL: align_fail={total_align_fail}, perf_ok={all_perf_ok}")
    if total_align_fail == 0 and all_perf_ok:
        print(f"  RESULT: ALL PASS")
    else:
        print(f"  RESULT: FAILURES DETECTED")
    print(f"{'#' * 70}")

    sys.exit(1 if total_align_fail > 0 else 0)


if __name__ == "__main__":
    main()
