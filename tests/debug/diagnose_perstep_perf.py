#!/usr/bin/env python3
"""Compare per-step decode throughput of kb-nano vs vLLM at FIXED batch sizes.

This isolates per-step kernel/scheduler efficiency from KV-cache concurrency
differences (e.g., vLLM packs 37% more decode sequences into the same
gpu_memory_utilization budget for DeepSeek-V3.2).

Methodology:
  * For each batch size BS in BS_LIST:
      - Run BS sequences with a tiny prompt (16 tokens) and a fixed decode
        length (DECODE).  All BS sequences enter decode together and stay
        together for DECODE-1 steps -> per-step batch == BS.
      - Time the second invocation (after warmup) to exclude first-step
        scheduling/JIT noise.
      - Report tokens/sec, ms/step, and ms/step/seq.
  * Both engines run in disposable subprocesses so memory is reset between
    them.

Usage:
    PYTHONPATH=/home/yak python -m kb_nano.tests.debug.diagnose_perstep_perf \\
        --model meta-llama/Llama-3.1-8B-Instruct --tp 1 \\
        --bs-list 16 32 64 128 256 512 --decode 64

For DeepSeek-V3.2 (large, slow to load) you probably want a smaller BS list:
    --bs-list 16 32 64 128 --decode 32
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = _REPO_ROOT
_PACKAGE_DIR = _REPO_ROOT / "kb_nano"


# ---------------------------------------------------------------------------
# vLLM worker: decode-only sweep at fixed batch sizes.
# ---------------------------------------------------------------------------
VLLM_WORKER = r'''
import json, os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

def main():
    from vllm import LLM, SamplingParams

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    llm = LLM(
        model=cfg["model"],
        tensor_parallel_size=cfg["tp"],
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
        seed=42,
    )

    # ---- Probe internal config so we can compare apples-to-apples.
    sched_cfg = llm.llm_engine.vllm_config.scheduler_config
    cache_cfg = llm.llm_engine.vllm_config.cache_config
    compile_cfg = llm.llm_engine.vllm_config.compilation_config
    cfg_dump = {
        "max_num_seqs": sched_cfg.max_num_seqs,
        "max_num_batched_tokens": sched_cfg.max_num_batched_tokens,
        "max_model_len": llm.llm_engine.vllm_config.model_config.max_model_len,
        "block_size": cache_cfg.block_size,
        "gpu_memory_utilization": cache_cfg.gpu_memory_utilization,
        "kv_cache_dtype": cache_cfg.cache_dtype,
        "compile_sizes": list(compile_cfg.cudagraph_capture_sizes or []),
        "compile_ranges_endpoints": list(getattr(compile_cfg, "compile_ranges_endpoints", []) or []),
        "cudagraph_mode": str(getattr(compile_cfg, "cudagraph_mode", "?")),
        "level": getattr(compile_cfg, "level", "?"),
        "pass_config": str(getattr(compile_cfg, "pass_config", "?")),
    }
    if hasattr(llm.llm_engine, "kv_cache_config"):
        kvc = llm.llm_engine.kv_cache_config
        cfg_dump["num_gpu_blocks"] = getattr(kvc, "num_blocks", None)
    elif hasattr(llm.llm_engine, "model_executor"):
        try:
            cfg_dump["num_gpu_blocks"] = llm.llm_engine.model_executor.driver_worker.gpu_blocks_total
        except Exception:
            pass

    # Warmup once to trigger any lazy init.
    llm.generate(
        [{"prompt_token_ids": [0] * 16}],
        SamplingParams(temperature=0.0, max_tokens=8),
    )

    decode = cfg["decode"]
    prompt_len = cfg["prompt_len"]
    results = []
    for bs in cfg["bs_list"]:
        prompts = [{"prompt_token_ids": [i % 1000 + 1] * prompt_len}
                   for i in range(bs)]
        sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=decode)

        # Warmup at this BS.
        llm.generate(prompts, sp, use_tqdm=False)

        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        out_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
        results.append({
            "bs": bs,
            "elapsed": elapsed,
            "out_tokens": out_toks,
            "tok_per_s": out_toks / elapsed,
            "ms_per_step": (elapsed / decode) * 1000.0,
            "ms_per_step_per_seq": (elapsed / decode / bs) * 1000.0,
        })

    del llm
    with open(cfg["output_file"], "w") as f:
        json.dump({"config": cfg_dump, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano worker: decode-only sweep at fixed batch sizes.
# ---------------------------------------------------------------------------
KB_WORKER = r'''
import json, os, sys, time

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])

    mod = __import__(f"{cfg['package_name']}.infra.engine",
                     fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine = LlamaEngine(
        model_name=cfg["model"],
        tensor_parallel_size=cfg["tp"],
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        max_model_len=cfg["max_model_len"],
        seed=42,
    )

    mr = engine.model_runner
    cfg_dump = {
        "max_num_seqs": getattr(mr, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(mr, "max_num_batched_tokens", None),
        "max_model_len": getattr(mr, "max_model_len", None),
        "num_blocks": getattr(mr, "num_blocks", None),
        "block_size": getattr(mr, "block_size", None),
        "kv_cache_token_slots": (
            (getattr(mr, "num_blocks", 0) or 0)
            * (getattr(mr, "block_size", 0) or 1)
        ),
        "graph_bs_list": list(getattr(mr, "graph_bs_list", [])),
        "enforce_eager": getattr(mr, "enforce_eager", None),
        "tp": getattr(engine, "tensor_parallel_size", None),
    }

    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=8))

    import torch
    decode = cfg["decode"]
    prompt_len = cfg["prompt_len"]
    results = []
    for bs in cfg["bs_list"]:
        # prompt as token ids matching vLLM input
        prompts = [[i % 1000 + 1] * prompt_len for i in range(bs)]
        sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=decode)

        engine.block_manager.reset()
        engine.generate(prompts, sp, use_tqdm=False)

        engine.block_manager.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = engine.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        out_toks = sum(len(o.token_ids) for o in outputs)
        results.append({
            "bs": bs,
            "elapsed": elapsed,
            "out_tokens": out_toks,
            "tok_per_s": out_toks / elapsed,
            "ms_per_step": (elapsed / decode) * 1000.0,
            "ms_per_step_per_seq": (elapsed / decode / bs) * 1000.0,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"config": cfg_dump, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
'''


def run_worker(label: str, code: str, cfg: dict, log_path: Path,
               timeout: int) -> dict | None:
    """Run worker in a subprocess and return its parsed JSON output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_file = Path(tmpdir) / "cfg.json"
        out_file = Path(tmpdir) / "out.json"
        cfg["output_file"] = str(out_file)
        cfg_file.write_text(json.dumps(cfg))

        worker_script = Path(tmpdir) / "worker.py"
        worker_script.write_text(code)

        print(f"[diag] Running {label} (writing log to {log_path})", flush=True)
        with log_path.open("w") as logf:
            proc = subprocess.run(
                [sys.executable, str(worker_script), str(cfg_file)],
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        if proc.returncode != 0:
            print(f"[diag] {label} subprocess FAILED (rc={proc.returncode}); "
                  f"see {log_path}")
            return None
        if not out_file.exists():
            print(f"[diag] {label} produced no output file; see {log_path}")
            return None
        return json.loads(out_file.read_text())


def fmt_results(label: str, data: dict) -> None:
    print(f"\n--- {label} CONFIG ---")
    for k, v in data.get("config", {}).items():
        if isinstance(v, list) and len(v) > 12:
            v = f"{v[:6]}... (len={len(v)})"
        print(f"  {k}: {v}")
    print(f"\n--- {label} RESULTS ---")
    print(f"  {'BS':>5} {'tok/s':>10} {'ms/step':>9} {'ms/step/seq':>13} "
          f"{'elapsed(s)':>11}")
    for r in data.get("results", []):
        print(f"  {r['bs']:>5d} {r['tok_per_s']:>10.0f} {r['ms_per_step']:>9.2f}"
              f" {r['ms_per_step_per_seq']:>13.3f} {r['elapsed']:>11.2f}")


def fmt_compare(vllm_data: dict, kb_data: dict) -> None:
    print(f"\n--- COMPARISON ---")
    print(f"  {'BS':>5} {'vLLM tok/s':>11} {'kb-nano tok/s':>14} "
          f"{'speedup':>9}  {'vLLM ms/step':>13} {'kb ms/step':>12}")
    by_bs_v = {r["bs"]: r for r in vllm_data["results"]}
    by_bs_k = {r["bs"]: r for r in kb_data["results"]}
    for bs in sorted(set(by_bs_v) | set(by_bs_k)):
        v = by_bs_v.get(bs)
        k = by_bs_k.get(bs)
        if v is None or k is None:
            continue
        speedup = k["tok_per_s"] / v["tok_per_s"]
        print(f"  {bs:>5d} {v['tok_per_s']:>11.0f} {k['tok_per_s']:>14.0f} "
              f"{speedup:>8.2f}x  {v['ms_per_step']:>13.2f} "
              f"{k['ms_per_step']:>12.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--bs-list", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--decode", type=int, default=64,
                   help="Number of decode steps to time per BS.")
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Defaults to prompt_len + decode + slack.")
    p.add_argument("--output-dir", default="/home/yak/kb_nano/tests/logs")
    p.add_argument("--timeout", type=int, default=10800)
    p.add_argument("--skip-vllm", action="store_true")
    p.add_argument("--skip-kb", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = args.model.split("/")[-1]
    suffix = f"{short}_tp{args.tp}_diag"
    log_v = out_dir / f"{suffix}_vllm.log"
    log_k = out_dir / f"{suffix}_kb.log"

    max_model_len = args.max_model_len
    if max_model_len is None:
        max_model_len = max(256, args.prompt_len + args.decode + 16)

    print("=" * 72)
    print("  Per-step performance diagnostic: kb-nano vs vLLM")
    print("=" * 72)
    print(f"  Model         : {args.model}")
    print(f"  TP            : {args.tp}")
    print(f"  Batch sizes   : {args.bs_list}")
    print(f"  Prompt len    : {args.prompt_len}  (decode={args.decode})")
    print(f"  Max model len : {max_model_len}")
    print(f"  Logs          : {log_v.name}, {log_k.name}")
    print("=" * 72)

    common_cfg = dict(
        model=args.model, tp=args.tp, bs_list=args.bs_list,
        decode=args.decode, prompt_len=args.prompt_len,
        max_model_len=max_model_len,
    )

    vllm_data = None
    if not args.skip_vllm:
        vllm_data = run_worker("vLLM", VLLM_WORKER, dict(common_cfg),
                               log_v, args.timeout)
        if vllm_data is not None:
            fmt_results("vLLM", vllm_data)

    kb_data = None
    if not args.skip_kb:
        kb_cfg = dict(common_cfg,
                      project_root=str(_PROJECT_ROOT),
                      package_name=_PACKAGE_DIR.name)
        kb_data = run_worker("kb-nano", KB_WORKER, kb_cfg, log_k, args.timeout)
        if kb_data is not None:
            fmt_results("kb-nano", kb_data)

    if vllm_data is not None and kb_data is not None:
        fmt_compare(vllm_data, kb_data)

    print()


if __name__ == "__main__":
    main()
