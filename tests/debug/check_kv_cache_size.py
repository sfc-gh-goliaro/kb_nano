#!/usr/bin/env python3
"""Compare KV-cache budget between kb-nano and vLLM on the same model.

Runs both engines back-to-back in disposable subprocesses and prints the
allocated KV-cache size (number of token slots).  Used to validate that
kb-nano's vLLM-ported memory-profiling formula yields the same KV budget.

Usage:
    PYTHONPATH=/home/yak python -m kb_nano.tests.debug.check_kv_cache_size \
        --model deepseek-ai/DeepSeek-V3.2 --tp 8 --max-model-len 4352
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


VLLM_WORKER = r'''
import json, os, sys
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

def main():
    from vllm import LLM
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
    cache_cfg = llm.llm_engine.vllm_config.cache_config
    # Dig into v1 internals to read the memory profile breakdown.
    out = {
        "engine": "vllm",
        "num_gpu_blocks": cache_cfg.num_gpu_blocks,
        "block_size": cache_cfg.block_size,
        "token_slots": cache_cfg.num_gpu_blocks * cache_cfg.block_size,
    }
    try:
        # vLLM v1: the executor holds a list of workers; each worker has the
        # snapshot fields (set in determine_available_memory).  Use the first
        # worker as the rank-0 representative.
        engine = llm.llm_engine
        worker = None
        for attr_chain in (
            ("model_executor", "driver_worker"),
            ("model_executor", "workers", 0),
        ):
            try:
                obj = engine
                for a in attr_chain:
                    obj = obj[a] if isinstance(a, int) else getattr(obj, a)
                worker = obj
                break
            except Exception:
                continue
        if worker is not None:
            for k in (
                "init_snapshot",
                "non_torch_memory",
                "peak_activation_memory",
                "available_kv_cache_memory_bytes",
                "cudagraph_memory_estimate",
                "requested_memory",
            ):
                v = getattr(worker, k, None)
                if v is None:
                    continue
                if hasattr(v, "total_memory"):
                    out[f"{k}_total_memory_bytes"] = v.total_memory
                    out[f"{k}_free_memory_bytes"] = v.free_memory
                else:
                    out[k + "_bytes"] = int(v)
            # weights memory lives on the model_runner
            try:
                wm = worker.model_runner.model_memory_usage
                out["weights_memory_bytes"] = int(wm)
            except Exception:
                pass
    except Exception as e:
        out["profile_dump_error"] = repr(e)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f)
    del llm

if __name__ == "__main__":
    main()
'''


KBNANO_WORKER = r'''
import json, os, sys
sys.path.insert(0, "''' + str(_REPO_ROOT) + r'''")

def main():
    from kb_nano.infra.engine import LlamaEngine
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    engine = LlamaEngine(
        model_name=cfg["model"],
        tensor_parallel_size=cfg["tp"],
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        max_model_len=cfg["max_model_len"],
    )
    mr = engine.model_runner
    is_mla = getattr(mr, "is_deepseek_mla", False)
    if is_mla:
        block_size = 64  # FlashMLA always uses block_size=64
    else:
        from kb_nano.infra.engine import BLOCK_SIZE
        block_size = BLOCK_SIZE
    num_blocks = mr.num_blocks
    out = {
        "engine": "kb-nano",
        "num_gpu_blocks": num_blocks,
        "block_size": block_size,
        "token_slots": num_blocks * block_size,
        "weights_bytes": mr._weights_memory,
        "torch_peak_increase_bytes": mr._torch_peak_increase,
        "non_torch_increase_bytes": mr._non_torch_increase,
        "total_memory_bytes": mr._baseline_snapshot.total_memory,
        "is_mla": is_mla,
    }
    with open(sys.argv[2], "w") as f:
        json.dump(out, f)
    del engine

if __name__ == "__main__":
    main()
'''


def run_engine(name: str, worker_src: str, cfg: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as wf:
        wf.write(worker_src)
        worker_path = wf.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as cf:
        json.dump(cfg, cf)
        cfg_path = cf.name
    out_path = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name

    print(f"\n[{name}] launching subprocess...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-u", worker_path, cfg_path, out_path],
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{name} subprocess failed (rc={proc.returncode})")
    with open(out_path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4352)
    p.add_argument("--engines", default="kb-nano,vllm")
    args = p.parse_args()

    cfg = {"model": args.model, "tp": args.tp, "max_model_len": args.max_model_len}
    engines = args.engines.split(",")
    results = {}
    if "kb-nano" in engines:
        results["kb-nano"] = run_engine("kb-nano", KBNANO_WORKER, cfg)
    if "vllm" in engines:
        results["vllm"] = run_engine("vllm", VLLM_WORKER, cfg)

    print("\n" + "=" * 72)
    print(f"KV cache size comparison for {args.model} (tp={args.tp}, mml={args.max_model_len})")
    print("=" * 72)
    for name, r in results.items():
        print(f"  {name:10s}: {r['num_gpu_blocks']} blocks x {r['block_size']} = "
              f"{r['token_slots']} token slots")
    if "kb-nano" in results and "vllm" in results:
        kb, v = results["kb-nano"]["token_slots"], results["vllm"]["token_slots"]
        ratio = kb / v if v else float("inf")
        print(f"  ratio kb-nano/vllm: {ratio:.3f} ({(ratio - 1) * 100:+.1f}%)")
    gib = 1 << 30
    if "kb-nano" in results:
        r = results["kb-nano"]
        print("\nkb-nano memory profile breakdown:")
        print(f"  total_memory          = {r['total_memory_bytes'] / gib:.2f} GiB")
        print(f"  weights               = {r['weights_bytes'] / gib:.2f} GiB")
        print(f"  peak_activation       = {r['torch_peak_increase_bytes'] / gib:.2f} GiB")
        print(f"  non_torch_increase    = {r['non_torch_increase_bytes'] / gib:.2f} GiB")
        non_kv = r['weights_bytes'] + r['torch_peak_increase_bytes'] + r['non_torch_increase_bytes']
        print(f"  non_kv_total          = {non_kv / gib:.2f} GiB")
        budget = r['total_memory_bytes'] * 0.9 - non_kv
        print(f"  computed kv budget    = {budget / gib:.2f} GiB (= total*0.9 - non_kv)")
    if "vllm" in results:
        r = results["vllm"]
        print("\nvllm memory profile breakdown:")
        for k, v in r.items():
            if k.endswith("_bytes"):
                print(f"  {k:40s} = {v / gib:.2f} GiB")
            elif k == "profile_dump_error":
                print(f"  profile_dump_error = {v}")


if __name__ == "__main__":
    main()
