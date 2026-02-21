#!/usr/bin/env python3
"""
Throughput benchmark: standalone engine vs vLLM vs sglang.

All engines run at full speed (CUDA graphs, torch.compile enabled) on the
same random-token-ID workload, matching the methodology from nano-vllm/bench.py.

Usage:
    # Default: 256 seqs, random 100-1024 input/output
    python tests/bench_throughput.py --model meta-llama/Llama-3.1-8B-Instruct

    # Large model with TP
    python tests/bench_throughput.py \
        --model meta-llama/Llama-3.1-70B-Instruct --tp 4

    # Quick sanity check
    python tests/bench_throughput.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --num-seqs 16 --max-input-len 256 --max-output-len 256
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from random import randint, seed as set_seed

# ---------------------------------------------------------------------------
# vLLM worker (runs in subprocess, full speed)
# ---------------------------------------------------------------------------
VLLM_WORKER = r'''
import json, os, sys, time

def main():
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    from vllm import LLM, SamplingParams

    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    llm = LLM(
        model=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=False,
        tensor_parallel_size=cfg["tp"],
    )

    prompt_token_ids = cfg["prompt_token_ids"]
    output_lens = cfg["output_lens"]

    sp_list = [
        SamplingParams(
            temperature=0.0, ignore_eos=True, max_tokens=ol,
        )
        for ol in output_lens
    ]

    # Warmup
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=16),
    )

    # Timed run
    vllm_prompts = [dict(prompt_token_ids=p) for p in prompt_token_ids]
    start = time.perf_counter()
    outputs = llm.generate(vllm_prompts, sp_list, use_tqdm=False)
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "elapsed": elapsed,
            "total_output_tokens": total_output_tokens,
        }, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Standalone worker (runs in subprocess, full speed)
# ---------------------------------------------------------------------------
STANDALONE_WORKER = r'''
import json, sys, time
with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    pkg = cfg["package_name"]
    mod = __import__(f"{pkg}.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine = LlamaEngine(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=False,
        tensor_parallel_size=cfg["tp"],
    )

    prompt_token_ids = cfg["prompt_token_ids"]
    output_lens = cfg["output_lens"]

    sp_list = [
        SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True)
        for ol in output_lens
    ]

    # Warmup
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    # Timed run
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompt_token_ids, sp_list)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(len(o.token_ids) for o in outputs)

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "elapsed": elapsed,
            "total_output_tokens": total_output_tokens,
        }, f)

    del engine

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# sglang worker (runs in subprocess, full speed)
# ---------------------------------------------------------------------------
SGLANG_WORKER = r'''
import json, os, sys, time

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=cfg["model"],
        tp_size=cfg["tp"],
        random_seed=cfg["seed"],
    )

    prompt_token_ids = cfg["prompt_token_ids"]
    output_lens = cfg["output_lens"]

    sp_list = [
        {"temperature": 0.0, "ignore_eos": True, "max_new_tokens": ol}
        for ol in output_lens
    ]

    # Warmup
    engine.generate(input_ids=[0] * 16, sampling_params={"temperature": 0.0, "max_new_tokens": 16})

    # Timed run
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(
        input_ids=prompt_token_ids,
        sampling_params=sp_list,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(
        o["meta_info"]["completion_tokens"]
        for o in outputs
    )

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "elapsed": elapsed,
            "total_output_tokens": total_output_tokens,
        }, f)

    engine.shutdown()

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def run_worker(script: str, config: dict, label: str) -> dict | None:
    """Run a worker script in a subprocess and return parsed JSON output."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp",
    ) as f:
        f.write(script)
        script_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp",
    ) as f:
        config["output_file"] = output_path
        json.dump(config, f)
        config_path = f.name

    try:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        result = subprocess.run(
            [sys.executable, script_path, config_path],
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"  ERROR: {label} failed with exit code {result.returncode}")
            return None

        with open(output_path) as f:
            return json.loads(f.read())
    finally:
        os.unlink(script_path)
        os.unlink(config_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Throughput benchmark: standalone engine vs vLLM",
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--num-seqs", type=int, default=256,
        help="Number of sequences (default: 256)",
    )
    parser.add_argument(
        "--max-input-len", type=int, default=1024,
        help="Max input length; each prompt is randint(100, L) tokens (default: 1024)",
    )
    parser.add_argument(
        "--max-output-len", type=int, default=1024,
        help="Max output length; each seq generates randint(100, L) tokens (default: 1024)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for workload generation (default: 0)",
    )
    parser.add_argument(
        "--save-vllm", type=str, default=None,
        help="Save vLLM results to a JSON file for reuse in later runs",
    )
    parser.add_argument(
        "--load-vllm", type=str, default=None,
        help="Load previously saved vLLM results instead of re-running vLLM",
    )
    parser.add_argument(
        "--save-sglang", type=str, default=None,
        help="Save sglang results to a JSON file for reuse in later runs",
    )
    parser.add_argument(
        "--load-sglang", type=str, default=None,
        help="Load previously saved sglang results instead of re-running sglang",
    )
    parser.add_argument(
        "--skip-sglang", action="store_true",
        help="Skip sglang benchmark",
    )
    parser.add_argument(
        "--skip-vllm", action="store_true",
        help="Skip vLLM benchmark",
    )
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(package_dir)
    package_name = os.path.basename(package_dir)

    # Generate workload (same random data for both engines)
    set_seed(args.seed)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    output_lens = [
        randint(100, args.max_output_len) for _ in range(args.num_seqs)
    ]
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_expected_output = sum(output_lens)

    short_name = args.model.split("/")[-1]

    print("=" * 70)
    print("  Throughput Benchmark: Ours vs vLLM vs sglang")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    print(f"  TP             : {args.tp}")
    print(f"  Sequences      : {args.num_seqs}")
    print(f"  Input lengths  : 100-{args.max_input_len} (total {total_input_tokens:,} tokens)")
    print(f"  Output lengths : 100-{args.max_output_len} (total {total_expected_output:,} tokens)")
    print(f"  Seed           : {args.seed}")
    print("=" * 70)

    config = {
        "model": args.model,
        "tp": args.tp,
        "seed": args.seed,
        "prompt_token_ids": prompt_token_ids,
        "output_lens": output_lens,
        "project_root": project_root,
        "package_name": package_name,
    }

    # --- Run vLLM ---
    if args.skip_vllm:
        vllm_data = None
    elif args.load_vllm:
        print(f"\n  Loading cached vLLM results from: {args.load_vllm}")
        with open(args.load_vllm) as f:
            vllm_data = json.load(f)
    else:
        vllm_data = run_worker(
            VLLM_WORKER, dict(config),
            f"vLLM  [{short_name}] (TP={args.tp}, full speed)",
        )
        if vllm_data and args.save_vllm:
            with open(args.save_vllm, "w") as f:
                json.dump(vllm_data, f, indent=2)
            print(f"  vLLM results saved to: {args.save_vllm}")

    # --- Run sglang ---
    if args.skip_sglang:
        sglang_data = None
    elif args.load_sglang:
        print(f"\n  Loading cached sglang results from: {args.load_sglang}")
        with open(args.load_sglang) as f:
            sglang_data = json.load(f)
    else:
        sglang_data = run_worker(
            SGLANG_WORKER, dict(config),
            f"sglang [{short_name}] (TP={args.tp}, full speed)",
        )
        if sglang_data and args.save_sglang:
            with open(args.save_sglang, "w") as f:
                json.dump(sglang_data, f, indent=2)
            print(f"  sglang results saved to: {args.save_sglang}")

    # --- Run Ours ---
    standalone_data = run_worker(
        STANDALONE_WORKER, dict(config),
        f"Ours  [{short_name}] (TP={args.tp}, full speed)",
    )

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")

    if vllm_data is None and standalone_data is None and sglang_data is None:
        print("  ERROR: All engines failed.")
        sys.exit(1)

    if vllm_data:
        v_tps = vllm_data["total_output_tokens"] / vllm_data["elapsed"]
        print(f"  vLLM:")
        print(f"    Output tokens : {vllm_data['total_output_tokens']:,}")
        print(f"    Time          : {vllm_data['elapsed']:.2f}s")
        print(f"    Throughput    : {v_tps:,.1f} output tok/s")
    else:
        v_tps = None
        if not args.skip_vllm:
            print("  vLLM: FAILED")

    if sglang_data:
        sg_tps = sglang_data["total_output_tokens"] / sglang_data["elapsed"]
        print(f"  sglang:")
        print(f"    Output tokens : {sglang_data['total_output_tokens']:,}")
        print(f"    Time          : {sglang_data['elapsed']:.2f}s")
        print(f"    Throughput    : {sg_tps:,.1f} output tok/s")
    else:
        sg_tps = None
        if not args.skip_sglang:
            print("  sglang: FAILED")

    if standalone_data:
        s_tps = standalone_data["total_output_tokens"] / standalone_data["elapsed"]
        print(f"  Ours:")
        print(f"    Output tokens : {standalone_data['total_output_tokens']:,}")
        print(f"    Time          : {standalone_data['elapsed']:.2f}s")
        print(f"    Throughput    : {s_tps:,.1f} output tok/s")
    else:
        s_tps = None
        print("  Ours: FAILED")

    print()
    if v_tps and s_tps:
        ratio = s_tps / v_tps
        winner = "Ours" if ratio > 1.0 else "vLLM"
        print(f"  Ours / vLLM   : {ratio:.2f}x  ({winner} is faster)")
    if sg_tps and s_tps:
        ratio = s_tps / sg_tps
        winner = "Ours" if ratio > 1.0 else "sglang"
        print(f"  Ours / sglang : {ratio:.2f}x  ({winner} is faster)")
    if v_tps and sg_tps:
        ratio = sg_tps / v_tps
        winner = "sglang" if ratio > 1.0 else "vLLM"
        print(f"  sglang / vLLM : {ratio:.2f}x  ({winner} is faster)")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
