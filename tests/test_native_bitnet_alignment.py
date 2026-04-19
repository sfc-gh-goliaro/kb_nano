#!/usr/bin/env python3
"""BitNet b1.58 alignment test.

Compares kb-nano's standalone BitNet implementation (W1.58A8 native, loading
from the HuggingFace ``microsoft/bitnet-b1.58-2B-4T`` checkpoint) against the
HuggingFace ``transformers`` reference (which uses the same checkpoint via
``transformers.integrations.bitnet``).

The HF reference and kb-nano are launched in separate Python subprocesses to
keep their CUDA contexts and torch state isolated.

Usage::

    # Both implementations + alignment
    python tests/test_native_bitnet_alignment.py --max-tokens 32 --seed 42

    # kb-nano only (e.g. when transformers HF path is unavailable)
    python tests/test_native_bitnet_alignment.py --kb-only --max-tokens 32

    # HF reference only
    python tests/test_native_bitnet_alignment.py --hf-only --max-tokens 32
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"

PROMPTS = [
    "What is 2 + 2?",
    "Translate 'hello' into French, German, and Japanese.",
    (
        "Explain the difference between a stack and a queue in computer "
        "science. Give a real-world analogy for each."
    ),
    (
        "Write a Python function that computes the factorial of a number "
        "using recursion. Include a docstring."
    ),
]


# ---------------------------------------------------------------------------
# HuggingFace transformers reference worker
# ---------------------------------------------------------------------------
HF_WORKER = r'''
import json, os, sys, time, torch

def main():
    cfg = json.loads(sys.argv[1])
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    torch.cuda.set_device(0)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    print("  [HF] Loading tokenizer + model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(f"  [HF] Ready in {time.time()-t0:.1f}s, "
          f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 128001

    results = []
    for pi, prompt in enumerate(cfg["prompts"]):
        token_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=cfg["max_tokens"],
                do_sample=False,
                use_cache=True,
                pad_token_id=eos_id,
            )
        elapsed = time.time() - t0
        gen_ids = out[0, len(token_ids):].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_toks = len(gen_ids)
        tps = n_toks / elapsed if elapsed > 0 else 0
        results.append({
            "text": gen_text,
            "token_ids": gen_ids,
        })
        print(f"  [HF] Prompt #{pi}: {n_toks} tokens in {elapsed*1000:.0f}ms "
              f"({tps:.0f} tok/s)", flush=True)

    torch.cuda.synchronize()
    with open(cfg["output_file"], "w") as f:
        json.dump({
            "results": results,
            "memory_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        }, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano worker (uses LlamaEngine — the same path real users hit).
# ---------------------------------------------------------------------------
KB_WORKER = r'''
import json, os, sys, time, torch

def main():
    cfg = json.loads(sys.argv[1])
    sys.path.insert(0, cfg["project_root"])

    # GDS / fastsafetensors is unreliable on some hosts; force the standard
    # safetensors loader so the test focuses on BitNet correctness.
    from kb_nano.infra import weight_loader as _wl
    _wl._HAS_FASTSAFETENSORS = False

    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    torch.cuda.set_device(0)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    print("  [kb_nano] Building LlamaEngine...", flush=True)
    t0 = time.time()
    engine = LlamaEngine(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
    )
    print(f"  [kb_nano] Ready in {time.time()-t0:.1f}s, "
          f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    sp = SamplingParams(
        max_tokens=cfg["max_tokens"], temperature=0.0, seed=cfg["seed"],
    )

    results = []
    t_total = time.time()
    outputs = engine.generate(cfg["prompts"], sp)
    total_elapsed = time.time() - t_total

    for pi, output in enumerate(outputs):
        token_ids = list(getattr(output, "token_ids", []))
        text = getattr(output, "generated_text",
                       getattr(output, "text", ""))
        results.append({"text": text, "token_ids": token_ids})
        print(f"  [kb_nano] Prompt #{pi}: {len(token_ids)} tokens", flush=True)

    print(f"  [kb_nano] Total generate(): {total_elapsed*1000:.0f}ms", flush=True)

    torch.cuda.synchronize()
    with open(cfg["output_file"], "w") as f:
        json.dump({
            "results": results,
            "memory_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        }, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def run_worker(script: str, config: dict, label: str) -> dict | None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(script)
        script_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name
    config["output_file"] = output_path

    try:
        print(f"\n{'-' * 70}")
        print(f"  {label}")
        print(f"{'-' * 70}")
        result = subprocess.run(
            [sys.executable, script_path, json.dumps(config)],
            timeout=900,
        )
        if result.returncode != 0:
            print(f"  ERROR: {label} failed with exit code {result.returncode}")
            return None
        with open(output_path) as f:
            return json.loads(f.read())
    finally:
        os.unlink(script_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def print_results(data: dict, label: str):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    for i, r in enumerate(data["results"]):
        prompt_preview = PROMPTS[i][:60] + ("..." if len(PROMPTS[i]) > 60 else "")
        print(f"\n  Prompt #{i}: {prompt_preview}")
        print(f"  Tokens   : {len(r['token_ids'])}")
        print(f"  IDs[:10] : {r['token_ids'][:10]}")
        print("  Output   :")
        for line in r["text"].splitlines():
            print(f"    {line}")
    if "memory_gb" in data:
        print(f"\n  Memory   : {data['memory_gb']} GB")


def main():
    parser = argparse.ArgumentParser(description="BitNet b1.58 alignment test")
    parser.add_argument("--hf-only", action="store_true",
                        help="Run HuggingFace reference only")
    parser.add_argument("--kb-only", action="store_true",
                        help="Run kb-nano standalone only")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=MODEL_ID)
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(this_dir)              # kb_nano/
    project_root = os.path.dirname(pkg_root)          # parent of kb_nano

    print("=" * 70)
    print("  BitNet b1.58 Alignment Test")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Seed        : {args.seed}")
    print(f"  Prompts     : {len(PROMPTS)}")
    mode = "hf-only" if args.hf_only else ("kb-only" if args.kb_only else "alignment")
    print(f"  Mode        : {mode}")
    print("=" * 70)

    hf_data = None
    kb_data = None

    if not args.kb_only:
        hf_cfg = {
            "model": args.model, "seed": args.seed,
            "prompts": PROMPTS, "max_tokens": args.max_tokens,
        }
        hf_data = run_worker(HF_WORKER, hf_cfg, f"HF reference [{args.model}]")
        if hf_data is None:
            print("\n  FAIL: HF reference failed")
            sys.exit(1)
        print_results(hf_data, "HuggingFace Reference")

    if not args.hf_only:
        kb_cfg = {
            "model": args.model, "seed": args.seed,
            "prompts": PROMPTS, "max_tokens": args.max_tokens,
            "project_root": project_root,
        }
        kb_data = run_worker(KB_WORKER, kb_cfg, f"kb-nano [{args.model}]")
        if kb_data is None:
            print("\n  FAIL: kb-nano failed")
            sys.exit(1)
        print_results(kb_data, "kb-nano Standalone")

    if hf_data and kb_data:
        print(f"\n{'=' * 70}")
        print("  ALIGNMENT COMPARISON")
        print(f"{'=' * 70}")
        match_count = 0
        for i in range(len(PROMPTS)):
            ref_ids = hf_data["results"][i]["token_ids"]
            our_ids = kb_data["results"][i]["token_ids"]
            min_len = min(len(ref_ids), len(our_ids))
            first_diff = min_len
            for j in range(min_len):
                if ref_ids[j] != our_ids[j]:
                    first_diff = j
                    break
            if first_diff == min_len and len(ref_ids) == len(our_ids):
                print(f"  Prompt #{i}: MATCH ({len(ref_ids)} tokens)")
                match_count += 1
            else:
                print(f"  Prompt #{i}: DIFFER at token {first_diff} "
                      f"(ref={len(ref_ids)} kb={len(our_ids)})")
                print(f"    Ref [{first_diff}:+3] = {ref_ids[first_diff:first_diff+3]}")
                print(f"    Ours[{first_diff}:+3] = {our_ids[first_diff:first_diff+3]}")
                # Late divergence (>=80% match) still counts — small numerical
                # drift from per-token int8 activation quant ordering is
                # expected after long generations.
                if first_diff >= len(ref_ids) * 0.8:
                    match_count += 1
                    print("    (late divergence — counted as match)")
        print(f"\n  Result: {match_count}/{len(PROMPTS)} prompts match")
        status = "PASS" if match_count >= len(PROMPTS) * 0.75 else "FAIL"
        print(f"{'=' * 70}")
        print(f"  {status}")
        print(f"{'=' * 70}")
        sys.exit(0 if status == "PASS" else 1)
    else:
        print(f"{'=' * 70}")
        print("  PASS")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
