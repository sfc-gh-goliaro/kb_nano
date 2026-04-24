#!/usr/bin/env python3
"""Kernel-level CUDA-time diff between kb-nano and vLLM at a FIXED decode batch.

This is a follow-up to `diagnose_perstep_perf.py`.  That script measured the
per-step wall-clock gap.  This one captures a torch.profiler trace around the
*same* steady-state decode workload and reports per-kernel CUDA self-time, then
diffs the two engines so we can see *which kernels* cost more in kb-nano.

Methodology:
  * For a single fixed batch size BS, prefill a tiny prompt (PROMPT_LEN=16) on
    BS sequences with `ignore_eos=True` and `max_tokens=DECODE`.  After two
    warmup `generate()` calls (which trigger CUDA-graph capture + DeepGEMM
    JIT for these shapes), we wrap a single `generate()` in torch.profiler.
  * For vLLM TP>1, GPU work runs in worker subprocesses, so we use vLLM's
    own profiler API (`LLM(profiler_config=...)` + `start_profile/stop_profile`)
    which fans out a torch.profiler instance into each TP worker.  We then
    read back rank-0's chrome trace and aggregate.
  * For kb-nano TP>1, rank 0 stays in the main process, so torch.profiler
    in the worker script captures rank-0 GPU activity directly.
  * The summary aggregates per-kernel:
      - name (CUDA kernel name as captured by Kineto)
      - count
      - self_cuda_us  (sum of dur across all invocations)
  * The driver bucketizes kernels (NCCL, attention, GEMM, MoE, RMSNorm, etc.)
    and prints a side-by-side delta.

Usage:
    PYTHONPATH=/home/yak python -m kb_nano.tests.debug.profile_kernels_diff \\
        --model deepseek-ai/DeepSeek-V3.2 --tp 8 --bs 128 --decode 32

Quick smoke test on Llama-8B TP=1:
    PYTHONPATH=/home/yak python -m kb_nano.tests.debug.profile_kernels_diff \\
        --model meta-llama/Llama-3.1-8B-Instruct --tp 1 --bs 128 --decode 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = _REPO_ROOT
_PACKAGE_DIR = _REPO_ROOT / "kb_nano"


# ---------------------------------------------------------------------------
# Bucketing rules.  Order matters: first match wins.
# Each entry is (bucket_name, regex), matched against lower-cased kernel name.
# ---------------------------------------------------------------------------
BUCKET_RULES_SRC: list[tuple[str, str]] = [
    # Comms
    ("nccl", r"nccl|all_reduce|allreduce|reduce_scatter|all_gather"),
    ("flashinfer_ar", r"flashinfer.*allreduce|trtllm.*allreduce|allreduce_fusion"),
    # Attention
    ("attention_mla", r"flash_mla|fmla_|flashmla"),
    ("attention_dense", r"flash_attn|fmha|flash_fwd"),
    ("sparse_indexer", r"fp8_mqa|fp8_paged_mqa|mqa_logits|topk_per_row|sparse_attn_indexer"),
    # GEMM
    ("gemm_fp8_deepgemm", r"deep_gemm|fp8_gemm_nt|fp8_blockwise|gemm_grouped|gemm_fp8|m_grouped_gemm"),
    ("gemm_bf16_cublas", r"cutlass|cublas|gemm.*bf16|gemm.*fp16|sm90.*gemm|hgemm|nvjet|splitkreduce"),
    # MoE
    ("moe_dispatch", r"fused_moe|fused_experts|moe_align|moe_sum|grouped_topk|grouped_softmax"),
    ("moe_router", r"router|topk|noaux"),
    # Norm + quant fusions
    ("norm_quant_fused", r"fused.*norm.*quant|fused.*rms.*quant|norm_quant|rms_quant"),
    ("rmsnorm", r"rms_norm|rmsnorm|fused_add_rms"),
    ("quant_per_token", r"per_token_group_quant|per_token_quant|act_quant|fp8_quant"),
    # SiLU/activation
    ("act_silu", r"silu|gelu|swiglu"),
    # Sampling / lm_head
    ("sampling", r"sample|argmax|multinomial|softmax"),
    # Memory ops
    ("copy_memcpy", r"memcpy|memset|copy_kernel|d2h|h2d"),
    # Inductor-generated triton (catch-all)
    ("triton_fused", r"triton_fused|triton_per_fused|triton_red"),
    # Elementwise
    ("elementwise", r"elementwise|vectorized|pointwise|index_kernel|gather|scatter"),
]


# ---------------------------------------------------------------------------
# vLLM worker: profile decode at fixed BS using vLLM's own profiler API.
# ---------------------------------------------------------------------------
VLLM_WORKER = r'''
import gzip, json, os, sys, time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    trace_dir = Path(cfg["trace_dir"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    # vLLM creates one .pt.trace.json[.gz] per worker.  Wipe stale traces so
    # we only see this run's outputs.
    for p in trace_dir.iterdir():
        try:
            p.unlink()
        except Exception:
            pass

    from vllm import LLM, SamplingParams
    import torch

    llm = LLM(
        model=cfg["model"],
        tensor_parallel_size=cfg["tp"],
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
        seed=42,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(trace_dir),
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": True,
            "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": True,
        },
    )

    bs = cfg["bs"]
    decode = cfg["decode"]
    prompt_len = cfg["prompt_len"]
    prompts = [{"prompt_token_ids": [i % 1000 + 1] * prompt_len} for i in range(bs)]
    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=decode)

    # ---- Warmup: triggers CUDA graph capture + DeepGEMM JIT for these shapes.
    llm.generate(prompts, sp, use_tqdm=False)
    llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()

    # ---- Profile a single steady-state generate().
    llm.start_profile()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    llm.stop_profile()

    # vLLM writes traces asynchronously when stop_profile() is called.  Give
    # workers a generous window to finish flushing to disk before we read.
    time.sleep(5.0)

    out_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
    summary = {
        "engine": "vllm",
        "bs": bs,
        "decode": decode,
        "prompt_len": prompt_len,
        "out_tokens": out_toks,
        "elapsed_s": elapsed,
        "trace_dir": str(trace_dir),
    }

    del llm
    with open(cfg["output_file"], "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# kb-nano worker: profile decode at fixed BS using torch.profiler in-process
# (rank 0 stays in main process, so we capture all rank-0 GPU work).
# ---------------------------------------------------------------------------
KB_WORKER = r'''
import json, os, sys, time
from pathlib import Path

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])

    import torch
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

    bs = cfg["bs"]
    decode = cfg["decode"]
    prompt_len = cfg["prompt_len"]
    prompts = [[i % 1000 + 1] * prompt_len for i in range(bs)]
    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=decode)

    # ---- Warmup.
    engine.block_manager.reset()
    engine.generate(prompts, sp, use_tqdm=False)
    engine.block_manager.reset()
    engine.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()

    # ---- Profile.
    engine.block_manager.reset()
    activities = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
    trace_file = cfg["trace_file"]
    Path(trace_file).parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        t0 = time.perf_counter()
        outputs = engine.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    prof.export_chrome_trace(trace_file)

    out_toks = sum(len(o.token_ids) for o in outputs)
    summary = {
        "engine": "kb_nano",
        "bs": bs,
        "decode": decode,
        "prompt_len": prompt_len,
        "out_tokens": out_toks,
        "elapsed_s": elapsed,
        "trace_file": trace_file,
    }

    with open(cfg["output_file"], "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Chrome trace parsing.
# ---------------------------------------------------------------------------
def _open_trace(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_chrome_trace(path: Path) -> dict:
    """Aggregate CUDA kernel events from a chrome trace JSON file.

    Returns: {name: {"count": N, "self_cuda_us": float}}.
    Considers events with cat in {"kernel","gpu_memcpy","gpu_memset"}.
    """
    by_name: dict[str, dict] = {}
    with _open_trace(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    for evt in events:
        cat = (evt.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        if (evt.get("ph") or "") != "X":
            continue
        dur = float(evt.get("dur") or 0.0)
        if dur <= 0:
            continue
        name = evt.get("name") or "<unnamed>"
        d = by_name.setdefault(name, {"count": 0, "self_cuda_us": 0.0})
        d["count"] += 1
        d["self_cuda_us"] += dur
    return by_name


def find_rank0_trace(trace_dir: Path) -> Path | None:
    """Heuristic: pick the largest trace file as rank 0 (it dominates GPU work
    in TP=1; for TP>1 vLLM emits one file per worker -- pick the first one
    matching local rank 0 if filename contains rank info, else the largest)."""
    files = [p for p in trace_dir.iterdir()
             if p.suffix in (".json", ".gz")
             or str(p).endswith(".json.gz")
             or str(p).endswith(".pt.trace.json")
             or str(p).endswith(".pt.trace.json.gz")]
    if not files:
        return None
    # Prefer files whose name contains 'rank0', 'rank_0', 'tp0', or '_0_'.
    for p in files:
        nm = p.name.lower()
        if any(tok in nm for tok in ("rank0", "rank_0", "_tp0", "_0_", "_pp0_tp0")):
            return p
    # Otherwise return the largest.
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files[0]


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def run_worker(label: str, code: str, cfg: dict, log_path: Path,
               timeout: int) -> dict | None:
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


def by_kernel_to_list(by_name: dict) -> list[dict]:
    rows = [{"name": n, **v} for n, v in by_name.items()]
    rows.sort(key=lambda r: r["self_cuda_us"], reverse=True)
    return rows


def bucket_for(name: str, rules: list[tuple[str, re.Pattern]]) -> str:
    n = name.lower()
    for bucket, pat in rules:
        if pat.search(n):
            return bucket
    return "other"


def aggregate_buckets(by_kernel: list[dict],
                      rules: list[tuple[str, re.Pattern]]) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for k in by_kernel:
        b = bucket_for(k["name"], rules)
        d = buckets.setdefault(b, {"count": 0, "self_cuda_us": 0.0,
                                   "examples": []})
        d["count"] += k["count"]
        d["self_cuda_us"] += k["self_cuda_us"]
        if len(d["examples"]) < 3:
            d["examples"].append(k["name"])
    return buckets


def fmt_top_kernels(label: str, by_kernel: list[dict], top: int) -> None:
    print(f"\n--- {label} TOP {top} KERNELS (by self CUDA us) ---")
    print(f"  {'self ms':>10} {'count':>8} {'us/call':>10}  name")
    print(f"  {'-' * 10} {'-' * 8} {'-' * 10}  {'-' * 60}")
    for k in by_kernel[:top]:
        per_call = k["self_cuda_us"] / max(1, k["count"])
        print(f"  {k['self_cuda_us']/1000:>10.2f} {k['count']:>8d} "
              f"{per_call:>10.2f}  {k['name'][:80]}")


def fmt_buckets(label: str, by_kernel: list[dict],
                rules: list[tuple[str, re.Pattern]]) -> None:
    buckets = aggregate_buckets(by_kernel, rules)
    total = sum(b["self_cuda_us"] for b in buckets.values()) or 1.0
    print(f"\n--- {label} BUCKETS (sorted by self CUDA us) ---")
    print(f"  {'bucket':<22} {'count':>8} {'self ms':>10} {'pct':>6}  examples")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 10} {'-' * 6}  {'-' * 40}")
    rows = sorted(buckets.items(), key=lambda kv: kv[1]["self_cuda_us"],
                  reverse=True)
    for name, d in rows:
        pct = d["self_cuda_us"] / total * 100.0
        ex = ", ".join(s[:32] for s in d["examples"])
        print(f"  {name:<22} {d['count']:>8d} {d['self_cuda_us']/1000:>10.2f}"
              f" {pct:>5.1f}%  {ex}")
    print(f"  {'TOTAL':<22} {'':>8} {total/1000:>10.2f} {100.0:>5.1f}%")


def fmt_bucket_diff(vllm_kernels: list[dict], kb_kernels: list[dict],
                    rules: list[tuple[str, re.Pattern]],
                    vllm_elapsed: float, kb_elapsed: float) -> None:
    bv = aggregate_buckets(vllm_kernels, rules)
    bk = aggregate_buckets(kb_kernels, rules)
    total_v = sum(b["self_cuda_us"] for b in bv.values()) or 1.0
    total_k = sum(b["self_cuda_us"] for b in bk.values()) or 1.0

    print(f"\n--- BUCKET DIFF (kb-nano - vLLM, sorted by absolute delta) ---")
    print(f"  vLLM   wall: {vllm_elapsed*1000:>8.2f} ms   GPU self: {total_v/1000:>8.2f} ms")
    print(f"  kb     wall: {kb_elapsed*1000:>8.2f} ms   GPU self: {total_k/1000:>8.2f} ms")
    print(f"  delta wall : {(kb_elapsed-vllm_elapsed)*1000:+.2f} ms"
          f"   GPU: {(total_k-total_v)/1000:+.2f} ms"
          f"  ({(total_k/total_v - 1)*100:+.1f}%)")

    rows = []
    all_buckets = set(bv) | set(bk)
    for b in all_buckets:
        v_self = bv.get(b, {}).get("self_cuda_us", 0.0)
        k_self = bk.get(b, {}).get("self_cuda_us", 0.0)
        v_ct = bv.get(b, {}).get("count", 0)
        k_ct = bk.get(b, {}).get("count", 0)
        rows.append((b, v_self, k_self, k_self - v_self, v_ct, k_ct))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)

    print(f"\n  {'bucket':<22} {'vLLM ms':>9} {'kb ms':>9} {'delta ms':>10}"
          f"  {'kb/vLLM':>8}  {'vLLM ct':>8} {'kb ct':>8}")
    print(f"  {'-' * 22} {'-' * 9} {'-' * 9} {'-' * 10}  {'-' * 8}  "
          f"{'-' * 8} {'-' * 8}")
    for b, v_self, k_self, delta, v_ct, k_ct in rows:
        ratio = (k_self / v_self) if v_self > 0 else float("inf")
        ratio_s = f"{ratio:>7.2f}x" if v_self > 0 else "    inf"
        print(f"  {b:<22} {v_self/1000:>9.2f} {k_self/1000:>9.2f}"
              f" {delta/1000:>+10.2f}  {ratio_s}"
              f"  {v_ct:>8d} {k_ct:>8d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--bs", type=int, default=128,
                   help="Decode batch size to profile.")
    p.add_argument("--decode", type=int, default=32,
                   help="Decode steps per profile run.")
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--output-dir", default="/home/yak/kb_nano/tests/logs")
    p.add_argument("--timeout", type=int, default=10800)
    p.add_argument("--skip-vllm", action="store_true")
    p.add_argument("--skip-kb", action="store_true")
    p.add_argument("--top", type=int, default=25,
                   help="Top-N kernels to print per engine.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = args.model.split("/")[-1]
    suffix = f"{short}_tp{args.tp}_bs{args.bs}_kprof"
    log_v = out_dir / f"{suffix}_vllm.log"
    log_k = out_dir / f"{suffix}_kb.log"
    json_v = out_dir / f"{suffix}_vllm.json"
    json_k = out_dir / f"{suffix}_kb.json"
    trace_dir_v = out_dir / f"{suffix}_vllm_traces"
    trace_file_k = out_dir / f"{suffix}_kb.trace.json"

    max_model_len = args.max_model_len
    if max_model_len is None:
        max_model_len = max(256, args.prompt_len + args.decode + 16)

    print("=" * 72)
    print("  Kernel-level CUDA-time diff: kb-nano vs vLLM")
    print("=" * 72)
    print(f"  Model         : {args.model}")
    print(f"  TP            : {args.tp}")
    print(f"  BS            : {args.bs}  (prompt_len={args.prompt_len}, decode={args.decode})")
    print(f"  Max model len : {max_model_len}")
    print(f"  Logs          : {log_v.name}, {log_k.name}")
    print(f"  Summaries     : {json_v.name}, {json_k.name}")
    print("=" * 72, flush=True)

    rules = [(b, re.compile(p)) for b, p in BUCKET_RULES_SRC]

    common_cfg = dict(
        model=args.model, tp=args.tp, bs=args.bs, decode=args.decode,
        prompt_len=args.prompt_len, max_model_len=max_model_len,
    )

    vllm_summary = None
    vllm_kernels: list[dict] = []
    if not args.skip_vllm:
        cfg_v = dict(common_cfg, trace_dir=str(trace_dir_v))
        vllm_summary = run_worker("vLLM", VLLM_WORKER, cfg_v, log_v, args.timeout)
        if vllm_summary is not None:
            trace_path = find_rank0_trace(trace_dir_v)
            if trace_path is None:
                print(f"[diag] vLLM produced no trace files in {trace_dir_v}")
            else:
                print(f"[diag] vLLM rank-0 trace: {trace_path.name} "
                      f"({trace_path.stat().st_size / 1e6:.1f} MB)")
                vllm_kernels = by_kernel_to_list(parse_chrome_trace(trace_path))
                vllm_summary["by_kernel"] = vllm_kernels
                json_v.write_text(json.dumps(vllm_summary, indent=2))
                fmt_top_kernels("vLLM", vllm_kernels, args.top)
                fmt_buckets("vLLM", vllm_kernels, rules)

    kb_summary = None
    kb_kernels: list[dict] = []
    if not args.skip_kb:
        cfg_k = dict(common_cfg, project_root=str(_PROJECT_ROOT),
                     package_name=_PACKAGE_DIR.name,
                     trace_file=str(trace_file_k))
        kb_summary = run_worker("kb-nano", KB_WORKER, cfg_k, log_k, args.timeout)
        if kb_summary is not None and trace_file_k.exists():
            print(f"[diag] kb-nano trace: {trace_file_k.name} "
                  f"({trace_file_k.stat().st_size / 1e6:.1f} MB)")
            kb_kernels = by_kernel_to_list(parse_chrome_trace(trace_file_k))
            kb_summary["by_kernel"] = kb_kernels
            json_k.write_text(json.dumps(kb_summary, indent=2))
            fmt_top_kernels("kb-nano", kb_kernels, args.top)
            fmt_buckets("kb-nano", kb_kernels, rules)

    if vllm_summary is not None and kb_summary is not None and \
            vllm_kernels and kb_kernels:
        fmt_bucket_diff(vllm_kernels, kb_kernels, rules,
                        vllm_summary["elapsed_s"], kb_summary["elapsed_s"])

    print()


if __name__ == "__main__":
    main()
