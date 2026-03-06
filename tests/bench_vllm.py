#!/usr/bin/env python3
"""
Throughput and alignment benchmark: kb-nano baseline vs vLLM.

Runs three scenarios (prefill-heavy, balanced, decode-heavy) with random
token IDs, compares throughput and per-token alignment.

Each engine (vLLM, kb-nano) is loaded once in a single long-lived subprocess
that processes all scenarios sequentially, avoiding repeated model loading.

Usage:
    python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct

    python tests/bench_vllm.py \
        --model meta-llama/Llama-3.1-70B-Instruct --tp 4

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

import numpy as np

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

# ---------------------------------------------------------------------------
# Multi-scenario vLLM subprocess worker
# ---------------------------------------------------------------------------
VLLM_WORKER = r'''
import json, sys, time

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

    with open(cfg["output_file"], "w") as f:
        json.dump(all_results, f)

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

    mod = __import__(f"{pkg}.engine", fromlist=["LlamaEngine", "SamplingParams"])
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
        outputs = engine.generate(prompts, sp_list)
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

    with open(cfg["output_file"], "w") as f:
        json.dump(all_results, f)

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
    parser.add_argument("--num-seqs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic alignment)",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument(
        "--no-enforce-eager", dest="enforce_eager", action="store_false",
    )
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save per-scenario outputs and results JSON "
             "(default: /tmp/bench_vllm/<model>_tp<tp>)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = f"/tmp/bench_vllm/{short}_tp{args.tp}"

    # Pre-generate all scenario data
    scenario_data = []
    global_max_seq_len = 0
    for i, scenario in enumerate(SCENARIOS):
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
            "input_len": input_len,
            "output_len": output_len,
            "prompt_token_ids": prompt_token_ids,
            "output_lens": output_lens,
        })

    print("=" * 70)
    print("  kb-nano Baseline vs vLLM -- Multi-Scenario Benchmark")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    print(f"  TP             : {args.tp}")
    print(f"  Seqs/scenario  : {args.num_seqs}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Enforce eager  : {args.enforce_eager}")
    print(f"  Seed           : {args.seed}")
    print(f"  Max seq len    : {global_max_seq_len}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Scenarios      : {', '.join(s['name'] for s in SCENARIOS)}")
    print("=" * 70)

    # -- Run vLLM (one subprocess, all scenarios) --
    vllm_results = None
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
        }
        vllm_results = run_worker(
            VLLM_WORKER, vllm_config,
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
    }
    short_name = args.model.split("/")[-1]
    kb_results = run_worker(
        KB_NANO_WORKER, kb_config,
        f"kb-nano [{short_name}] all scenarios (TP={args.tp})",
    )
    if kb_results is None:
        print("  ERROR: kb-nano subprocess failed.")
        sys.exit(1)

    # -- Compute metrics per scenario --
    all_results = []
    for i, scenario in enumerate(SCENARIOS):
        kb_data = kb_results[i]
        kb_tps = kb_data["total_output_tokens"] / kb_data["elapsed"]

        result = {
            "scenario": scenario["name"],
            "input_len": scenario["input_len"],
            "output_len": scenario["output_len"],
            "num_seqs": args.num_seqs,
            "kb_nano_elapsed": kb_data["elapsed"],
            "kb_nano_output_tokens": kb_data["total_output_tokens"],
            "kb_nano_tok_per_s": kb_tps,
        }

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

        # Save per-scenario outputs
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

    # -- Summary table --
    print(f"\n\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
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

        print(
            f"  {r['scenario']:<16} {r['input_len']:>5} {r['output_len']:>5} "
            f"{kb_tps_str:>15} {v_tps_str:>12} {speedup_str:>8} "
            f"{match_str:>15}"
        )

    print(f"{'=' * 90}")

    # -- Save combined results --
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "results.json")
        combined = {
            "model": args.model,
            "tp": args.tp,
            "seed": args.seed,
            "temperature": args.temperature,
            "num_seqs": args.num_seqs,
            "enforce_eager": args.enforce_eager,
            "scenarios": all_results,
        }
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
