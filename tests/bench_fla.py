#!/usr/bin/env python3
"""
Throughput, latency, and alignment benchmark: kb-nano FLAEngine vs FLA reference.

Mirrors the structure of ``tests/bench_vllm.py`` but for the recurrent
linear-attention models that ship in this branch:

  - GLA       (fla-hub/gla-2.7B-100B)
  - RetNet    (fla-hub/retnet-2.7B-100B)
  - RWKV7     (fla-hub/rwkv7-2.9B-g1)

The reference implementation is the SOTA ``flash-linear-attention``
library: ``fla.models.{gla,retnet,rwkv7}.{...}ForCausalLM`` driven via
``transformers.generate``, which is exactly the recipe FLA's own
``benchmarks/benchmark_generation.py`` uses. Both sides batch all
prompts in a single forward call so they exercise their respective
batched-decode paths apples-to-apples (kb-nano's FLAEngine adds
chunked prefill + continuous batching on top).

Each engine is launched in its own long-lived subprocess that processes
all scenarios sequentially, avoiding repeated model loading.

Usage:
    python tests/bench_fla.py --model fla-hub/gla-2.7B-100B
    python tests/bench_fla.py --model fla-hub/retnet-2.7B-100B
    python tests/bench_fla.py --model fla-hub/rwkv7-2.9B-g1
    python tests/bench_fla.py --model ... --skip-fla   # kb-nano only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from random import randint

import numpy as np


def _detect_gpu_name() -> str:
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
from kb_nano.tests.bench_vllm import compute_alignment


# Same scenario shapes as bench_vllm.py so the workloads are directly
# comparable to the dense LLM bench.
SCENARIOS = [
    {"name": "prefill-heavy", "input_len": 1024, "output_len": 512},
    {"name": "balanced",      "input_len": 512,  "output_len": 512},
    {"name": "decode-heavy",  "input_len": 512,  "output_len": 1024},
]

LATENCY_SCENARIOS = [
    {"name": "single-request", "input_len": 128, "output_len": 128, "batch_size": 1},
    {"name": "fixed-batch-32", "input_len": 128, "output_len": 128, "batch_size": 32},
]


# ---------------------------------------------------------------------------
# FLA reference subprocess worker
# ---------------------------------------------------------------------------
# Uses fla.models.{retnet,gla,rwkv7}.*ForCausalLM driven via transformers'
# .generate(). HF padding produces a [B, T] left-padded tensor and FLA's
# layers consume the corresponding past_key_values cache for batched decode.
FLA_REF_WORKER = r'''
import json, os, sys, time
import warnings

# Suppress one expected FLA warning during T=1 RWKV7 decode (heuristic check
# fires when seq_len < num_heads, which is normal for our shapes).
warnings.filterwarnings(
    "ignore",
    message=r".*seq_len.*<.*num_heads.*",
    category=UserWarning,
)


def _load_fla_model(model_name, device, dtype):
    from huggingface_hub import snapshot_download
    if "rwkv7" in model_name:
        from fla.models.rwkv7 import RWKV7ForCausalLM as M
    elif "retnet" in model_name:
        from fla.models.retnet import RetNetForCausalLM as M
    elif "gla" in model_name:
        from fla.models.gla import GLAForCausalLM as M
    else:
        raise ValueError(f"unknown FLA model family: {model_name!r}")
    path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    model = M.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    return model


def _batched_generate(model, tokenizer, prompts, max_tokens, eos, device,
                      ignore_eos=True):
    """Pad-batch prompts and run a single .generate() call.

    Returns list[list[int]] of generated token ids (excluding the prompt).
    """
    import torch
    from transformers import GenerationConfig

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        eos if eos is not None else 0
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_id

    # prompts is a list of token-id lists. Build a left-padded [B, T] tensor.
    max_len = max(len(p) for p in prompts)
    input_ids = torch.full(
        (len(prompts), max_len), pad_id, dtype=torch.long, device=device,
    )
    attn = torch.zeros(
        (len(prompts), max_len), dtype=torch.long, device=device,
    )
    for i, p in enumerate(prompts):
        input_ids[i, max_len - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        attn[i, max_len - len(p):] = 1

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
        pad_token_id=pad_id,
    )
    if not ignore_eos and eos is not None:
        gen_kwargs["eos_token_id"] = eos

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
    # Strip the prompt prefix from each row
    gen = out[:, max_len:]
    return [row.tolist() for row in gen]


def main():
    import torch
    from transformers import AutoTokenizer

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    device = "cuda"
    dtype = torch.bfloat16
    model_name = cfg["model"]

    print(f"  [FLA reference] loading {model_name}...", flush=True)
    model = _load_fla_model(model_name, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos = tokenizer.eos_token_id

    # Warmup
    _batched_generate(model, tokenizer, [[0] * 16], 16, eos, device, ignore_eos=True)
    torch.cuda.synchronize()

    scenarios = cfg["scenarios"]
    all_results = []
    for sc in scenarios:
        prompts = sc["prompt_token_ids"]
        out_lens = sc["output_lens"]
        max_out = max(out_lens) if out_lens else cfg.get("default_output_len", 128)

        # Process in micro-batches of `max_num_seqs` so peak memory is the
        # same as kb-nano's continuous-batching ceiling. HF's .generate
        # has no continuous-batching, so this is the apples-to-apples way
        # to keep both engines at the same concurrency.
        bs_cap = cfg.get("max_num_seqs", 256)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen_tokens = []
        for off in range(0, len(prompts), bs_cap):
            sub = prompts[off:off + bs_cap]
            gen_tokens.extend(_batched_generate(
                model, tokenizer, sub, max_out, eos, device, ignore_eos=True,
            ))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_in = sum(len(p) for p in prompts)
        total_out = sum(len(g) for g in gen_tokens)
        all_results.append({
            "name": sc["name"],
            "elapsed": elapsed,
            "total_prompt_tokens": total_in,
            "total_output_tokens": total_out,
            "outputs": [
                {"text": tokenizer.decode(g, skip_special_tokens=True),
                 "token_ids": g}
                for g in gen_tokens
            ],
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompt_token_ids"]
        out_len = ls["output_len"]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            _batched_generate(model, tokenizer, prompts, out_len, eos, device,
                              ignore_eos=True)
        torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _batched_generate(model, tokenizer, prompts, out_len, eos, device,
                              ignore_eos=True)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        latency_results.append({
            "name": ls["name"],
            "batch_size": ls["batch_size"],
            "input_len": ls["input_len"],
            "output_len": out_len,
            "num_iters": num_iters,
            "latencies": latencies,
        })

    del model

    with open(cfg["output_file"], "w") as f:
        json.dump({"throughput": all_results, "latency": latency_results}, f)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano FLAEngine subprocess worker
# ---------------------------------------------------------------------------
KB_NANO_FLA_WORKER = r'''
import json, os, sys, time


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    if cfg.get("pytorch_reference", False):
        from kb_nano.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    mod = __import__(
        f"{pkg}.infra.fla_engine",
        fromlist=["FLAEngine", "SamplingParams"],
    )
    FLAEngine, SamplingParams = mod.FLAEngine, mod.SamplingParams

    engine = FLAEngine(
        model_name=cfg["model"],
        seed=cfg["seed"],
        max_num_seqs=cfg.get("max_num_seqs", 256),
        chunked_prefill_size=cfg.get("chunked_prefill_size", 256),
    )

    # Warmup
    engine.generate([[0] * 16], SamplingParams(temperature=0.0, max_tokens=16,
                                               ignore_eos=True))

    import torch
    scenarios = cfg["scenarios"]
    all_results = []
    for sc in scenarios:
        prompts = sc["prompt_token_ids"]
        output_lens = sc["output_lens"]
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

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = engine.generate(prompts, sp_list, use_tqdm=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_in = sum(len(p) for p in prompts)
        total_out = sum(len(o.token_ids) for o in outputs)
        all_results.append({
            "name": sc["name"],
            "elapsed": elapsed,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "outputs": [
                {"generated_text": o.generated_text, "token_ids": o.token_ids}
                for o in outputs
            ],
        })

    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        prompts = ls["prompt_token_ids"]
        sp = SamplingParams(temperature=0.0, ignore_eos=True,
                            max_tokens=ls["output_len"])
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)
        for _ in range(num_warmup):
            engine.generate(prompts, sp)
            torch.cuda.synchronize()
        latencies = []
        for _ in range(num_iters):
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


SUPPORTED_MODELS = {
    "fla-hub/gla-2.7B-100B",
    "fla-hub/retnet-2.7B-100B",
    "fla-hub/rwkv7-2.9B-g1",
    "fla-hub/rwkv7-2.9B-world",
}


def main():
    parser = argparse.ArgumentParser(
        description="Throughput & alignment benchmark: kb-nano FLAEngine vs FLA reference",
    )
    parser.add_argument("--model", type=str, default="fla-hub/gla-2.7B-100B")
    parser.add_argument("--num-seqs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Max concurrent sequences in FLAEngine "
                             "(default tuned for the bench_vllm.py workload)")
    parser.add_argument("--chunked-prefill-size", type=int, default=1024,
                        help="Per-chunk prefill size (rounded up to a multiple "
                             "of 64). Default tuned for the bench_vllm.py workload.")
    parser.add_argument("--skip-fla", action="store_true",
                        help="Skip the FLA reference (kb-nano only)")
    parser.add_argument(
        "--pytorch-reference", action="store_true", default=False,
        help="Patch semantic PyTorch references from tasks/reference/L*/ into kb-nano.",
    )
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--latency-iters", type=int, default=3)
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run only the throughput scenario with this name "
             "(e.g. 'balanced'). Default: all scenarios.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.model not in SUPPORTED_MODELS:
        print(f"WARNING: {args.model!r} not in known list: {sorted(SUPPORTED_MODELS)}",
              file=sys.stderr)

    gpu = _detect_gpu_name()
    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = str(_PACKAGE_DIR / "tests" / "results" / gpu / f"{short}_fla_tp1")

    throughput_scenarios = list(SCENARIOS)
    latency_scenarios = list(LATENCY_SCENARIOS)
    if args.scenario is not None:
        throughput_scenarios = [
            s for s in throughput_scenarios if s["name"] == args.scenario
        ]
        if not throughput_scenarios:
            raise SystemExit(f"--scenario={args.scenario!r} did not match any scenario")

    # Pre-generate all prompts so both engines see identical inputs.
    scenario_data = []
    if not args.skip_throughput:
        for i, sc in enumerate(throughput_scenarios):
            rng = args.seed + i
            random.seed(rng)
            np.random.seed(rng)
            input_len = sc["input_len"]
            output_len = sc["output_len"]
            prompt_token_ids = [
                [randint(0, 10000) for _ in range(input_len)]
                for _ in range(args.num_seqs)
            ]
            scenario_data.append({
                "name": sc["name"],
                "input_len": input_len,
                "output_len": output_len,
                "prompt_token_ids": prompt_token_ids,
                "output_lens": [output_len] * args.num_seqs,
            })

    latency_data = []
    if not args.skip_latency:
        for j, ls in enumerate(latency_scenarios):
            rng = args.seed + 100 + j
            random.seed(rng)
            np.random.seed(rng)
            bs = ls["batch_size"]
            prompt_token_ids = [
                [randint(0, 10000) for _ in range(ls["input_len"])]
                for _ in range(bs)
            ]
            latency_data.append({
                "name": ls["name"],
                "input_len": ls["input_len"],
                "output_len": ls["output_len"],
                "batch_size": bs,
                "prompt_token_ids": prompt_token_ids,
                "num_warmup": 2,
                "num_iters": args.latency_iters,
            })

    print("=" * 70)
    print("  kb-nano FLAEngine vs FLA reference -- Multi-Scenario Benchmark")
    print("=" * 70)
    print(f"  Model            : {args.model}")
    print(f"  Seqs/scenario    : {args.num_seqs}")
    print(f"  max_num_seqs     : {args.max_num_seqs}")
    print(f"  chunk size       : {args.chunked_prefill_size}")
    print(f"  Temperature      : {args.temperature}")
    print(f"  GPU              : {gpu}")
    print(f"  Seed             : {args.seed}")
    print(f"  Output dir       : {args.output_dir}")
    if scenario_data:
        print(f"  Throughput       : {', '.join(s['name'] for s in throughput_scenarios)}")
    if latency_data:
        print(f"  Latency          : "
              f"{', '.join(s['name'] for s in latency_scenarios)} "
              f"({args.latency_iters} iters)")
    print("=" * 70)

    short_name = args.model.split("/")[-1]
    base_cfg = {
        "model": args.model,
        "seed": args.seed,
        "temperature": args.temperature,
        "scenarios": scenario_data,
        "latency_scenarios": latency_data,
    }

    fla_raw = None
    if not args.skip_fla:
        fla_cfg = dict(base_cfg)
        fla_cfg["max_num_seqs"] = args.max_num_seqs
        fla_raw = run_worker(
            FLA_REF_WORKER, fla_cfg,
            f"FLA reference [{short_name}] all scenarios",
            timeout=10800,
        )

    kb_cfg = dict(base_cfg)
    kb_cfg["project_root"] = str(_PROJECT_ROOT)
    kb_cfg["package_name"] = _PACKAGE_DIR.name
    kb_cfg["max_num_seqs"] = args.max_num_seqs
    kb_cfg["chunked_prefill_size"] = args.chunked_prefill_size
    kb_cfg["pytorch_reference"] = args.pytorch_reference
    kb_raw = run_worker(
        KB_NANO_FLA_WORKER, kb_cfg,
        f"kb-nano FLAEngine [{short_name}] all scenarios",
        timeout=10800,
    )
    if kb_raw is None:
        print("  ERROR: kb-nano subprocess failed.")
        sys.exit(1)

    kb_latency = kb_raw.get("latency", [])
    fla_latency = fla_raw.get("latency", []) if fla_raw else []

    # ------------------------------------------------------------------
    # Throughput summary
    # ------------------------------------------------------------------
    all_results = []
    if not args.skip_throughput:
        kb_thr = kb_raw["throughput"]
        fla_thr = fla_raw["throughput"] if fla_raw else None
        for i, sc in enumerate(throughput_scenarios):
            kb_d = kb_thr[i]
            kb_tps = kb_d["total_output_tokens"] / kb_d["elapsed"]
            r = {
                "scenario": sc["name"],
                "input_len": sc["input_len"],
                "output_len": sc["output_len"],
                "num_seqs": args.num_seqs,
                "kb_nano_elapsed": kb_d["elapsed"],
                "kb_nano_output_tokens": kb_d["total_output_tokens"],
                "kb_nano_tok_per_s": kb_tps,
            }
            if fla_thr is not None:
                f_d = fla_thr[i]
                f_tps = f_d["total_output_tokens"] / f_d["elapsed"]
                r["fla_elapsed"] = f_d["elapsed"]
                r["fla_output_tokens"] = f_d["total_output_tokens"]
                r["fla_tok_per_s"] = f_tps
                r["speedup"] = kb_tps / f_tps if f_tps else 0.0
                if args.temperature == 0.0:
                    r["alignment"] = compute_alignment(
                        kb_d["outputs"], f_d["outputs"]
                    )
            if args.output_dir:
                d = os.path.join(args.output_dir, sc["name"])
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "kb_nano_outputs.json"), "w") as f:
                    json.dump(kb_d, f, indent=2)
                if fla_thr is not None:
                    with open(os.path.join(d, "fla_outputs.json"), "w") as f:
                        json.dump(f_d, f, indent=2)
            all_results.append(r)

        print(f"\n\n{'=' * 100}")
        print("  THROUGHPUT SUMMARY (kb-nano FLAEngine vs FLA reference)")
        print(f"{'=' * 100}")
        print(
            f"  {'SCENARIO':<16} {'IN':>5} {'OUT':>5} "
            f"{'KB-NANO tok/s':>15} {'FLA tok/s':>12} {'SPEEDUP':>9} "
            f"{'AVG MATCH TOKS':>18}"
        )
        print(f"  {'-' * 95}")
        for r in all_results:
            kb_str = f"{r['kb_nano_tok_per_s']:,.0f}"
            f_str = f"{r['fla_tok_per_s']:,.0f}" if "fla_tok_per_s" in r else "N/A"
            spd_str = f"{r['speedup']:.2f}x" if "speedup" in r else "N/A"
            align = r.get("alignment", {})
            avg = align.get("avg_matching_tokens_per_request", 0)
            tot = align.get("avg_output_len", 0)
            match_str = f"{avg:.1f}/{tot:.0f}" if tot > 0 else "N/A"
            print(
                f"  {r['scenario']:<16} {r['input_len']:>5} {r['output_len']:>5} "
                f"{kb_str:>15} {f_str:>12} {spd_str:>9} {match_str:>18}"
            )
        print("=" * 100)

    # ------------------------------------------------------------------
    # Latency summary
    # ------------------------------------------------------------------
    latency_combined = []
    if kb_latency:
        print(f"\n{'=' * 110}")
        print("  LATENCY SUMMARY")
        print(f"{'=' * 110}")
        print(
            f"  {'SCENARIO':<18} {'BS':>4} {'OUT':>5} {'ITERS':>6}"
            f"  {'KB-NANO med':>12} {'FLA med':>12}"
            f"  {'KB-NANO ms/tok':>15} {'FLA ms/tok':>12} {'SPEEDUP':>8}"
        )
        print(f"  {'-' * 105}")

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

            f_med_str = "N/A"; spd_str = "N/A"; f_ms_str = "N/A"
            if i < len(fla_latency):
                f_lat = fla_latency[i]
                f_lats = np.array(f_lat["latencies"])
                f_med = float(np.median(f_lats))
                f_ms_per_tok = (f_med / total_out_tokens) * 1000
                spd = f_med / kb_med
                f_med_str = f"{f_med:.4f}s"
                spd_str = f"{spd:.2f}x"
                f_ms_str = f"{f_ms_per_tok:.2f}"
                lat_result["fla_median_s"] = f_med
                lat_result["fla_ms_per_tok"] = f_ms_per_tok
                lat_result["speedup"] = spd
                lat_result["fla_latencies"] = f_lat["latencies"]

            print(
                f"  {kb_lat['name']:<18} {bs:>4} {out_len:>5} {kb_lat['num_iters']:>6}"
                f"  {kb_med:.4f}s{'':<3} {f_med_str:>12}"
                f"  {kb_ms_per_tok:>13.2f}   {f_ms_str:>10} {spd_str:>8}"
            )
            latency_combined.append(lat_result)

        print("=" * 110)

    if args.output_dir and (all_results or latency_combined):
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({
                "gpu": gpu,
                "model": args.model,
                "seed": args.seed,
                "temperature": args.temperature,
                "num_seqs": args.num_seqs,
                "max_num_seqs": args.max_num_seqs,
                "chunked_prefill_size": args.chunked_prefill_size,
                "scenarios": all_results,
                "latency_scenarios": latency_combined,
            }, f, indent=2)
        print(f"\n  Results saved to: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
