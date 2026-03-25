#!/usr/bin/env python3
"""Operator-level profiling: kb-nano vs vLLM for DeepSeek-V3.2 (4 layers).

Profiles both frameworks to get GPU kernel timings for comparison.
- kb-nano: torch.profiler wrapping engine.generate()
- vLLM: built-in profiler via LLM.start_profile()/stop_profile()

Usage:
    cd /home/yak/kb_nano
    PYTHONUNBUFFERED=1 python tests/debug/profile_deepseek_ops.py > /tmp/profile_ops.log 2>&1
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

VLLM_PROFILER = r'''
import json, os, sys, time, gc, glob
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

import torch

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    profile_dir = cfg.get("profile_dir", "/tmp/vllm_profile")
    os.makedirs(profile_dir, exist_ok=True)

    from vllm import LLM, SamplingParams

    num_layers = cfg["num_layers"]
    enforce_eager = cfg.get("enforce_eager", False)

    llm = LLM(
        model=cfg["model"],
        seed=42,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
        load_format="auto",
        hf_overrides={"num_hidden_layers": num_layers},
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": profile_dir,
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_use_gzip": False,
        },
    )

    # Warmup
    llm.generate(
        [dict(prompt_token_ids=[0]*128)],
        SamplingParams(temperature=0.0, max_tokens=16),
    )

    results = {}

    for label, input_len, bs, max_tok in cfg["scenarios"]:
        prompts = [
            dict(prompt_token_ids=list(range(input_len)))
            for _ in range(bs)
        ]
        sp = SamplingParams(temperature=0.0, max_tokens=max_tok, ignore_eos=True)

        # Warmup this scenario
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        time.sleep(0.5)

        # Wall-clock timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1000
        total_tokens = bs * max_tok
        tok_per_s = total_tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0

        print(f"\n=== vLLM {label} ===")
        print(f"  Wall time: {wall_ms:.2f} ms")
        print(f"  Tokens: {total_tokens}")
        print(f"  Throughput: {tok_per_s:.0f} tok/s")
        sys.stdout.flush()

        results[label] = {
            "wall_ms": wall_ms,
            "total_tokens": total_tokens,
            "tok_per_s": tok_per_s,
        }

    # Profile the target scenario using vLLM's built-in profiler
    profile_label = cfg.get("profile_scenario", cfg["scenarios"][0][0])
    for label, input_len, bs, max_tok in cfg["scenarios"]:
        if label == profile_label:
            prompts = [
                dict(prompt_token_ids=list(range(input_len)))
                for _ in range(bs)
            ]
            sp = SamplingParams(temperature=0.0, max_tokens=max_tok, ignore_eos=True)

            # Warmup
            llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            time.sleep(0.5)

            # Clear old profile files
            for f in glob.glob(os.path.join(profile_dir, "*.json*")):
                os.unlink(f)

            # Start built-in profiler
            llm.start_profile()

            # Run the scenario
            llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()

            # Stop profiler (triggers trace dump)
            llm.stop_profile()

            print(f"\n=== vLLM {label} (profiled) ===")
            print(f"  Profile traces saved to: {profile_dir}")

            # Find and load the trace file from the worker
            trace_files = sorted(glob.glob(os.path.join(profile_dir, "*.json*")))
            for tf in trace_files:
                print(f"  Trace file: {tf} ({os.path.getsize(tf)} bytes)")

            # Load the worker trace (not the async_llm one) and print kernel summary
            for tf in trace_files:
                if "EngineCore" in tf or "worker" in tf.lower():
                    try:
                        with open(tf) as fh:
                            trace_data = json.load(fh)
                        events = trace_data.get("traceEvents", [])
                        kernel_events = [
                            e for e in events
                            if e.get("cat") == "kernel"
                        ]
                        print(f"\n  Worker trace: {os.path.basename(tf)}")
                        print(f"  Total kernel events: {len(kernel_events)}")

                        # Aggregate by kernel name
                        kernel_times = {}
                        for e in kernel_events:
                            name = e.get("name", "unknown")
                            dur_us = e.get("dur", 0)
                            if name not in kernel_times:
                                kernel_times[name] = {"count": 0, "total_us": 0}
                            kernel_times[name]["count"] += 1
                            kernel_times[name]["total_us"] += dur_us

                        total_gpu_us = sum(v["total_us"] for v in kernel_times.values())

                        # Sort by total time
                        sorted_kernels = sorted(
                            kernel_times.items(),
                            key=lambda x: x[1]["total_us"],
                            reverse=True,
                        )

                        print(f"  Total GPU kernel time: {total_gpu_us/1000:.2f} ms")
                        print(f"\n  {'Kernel':<80} {'Count':>6} {'Total ms':>10} {'%':>6}")
                        print(f"  {'-'*80} {'-'*6} {'-'*10} {'-'*6}")
                        for name, info in sorted_kernels[:40]:
                            short_name = name[:80]
                            pct = info["total_us"] / total_gpu_us * 100 if total_gpu_us > 0 else 0
                            print(f"  {short_name:<80} {info['count']:>6} {info['total_us']/1000:>10.3f} {pct:>5.1f}%")

                    except Exception as ex:
                        print(f"  Error reading trace: {ex}")

            sys.stdout.flush()
            break

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
'''

KB_PROFILER = r'''
import json, os, sys, time, gc

with open(sys.argv[1]) as f:
    cfg = json.load(f)

os.environ["KB_NANO_NUM_LAYERS"] = str(cfg["num_layers"])
sys.path.insert(0, cfg["project_root"])

import torch

def main():
    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    enforce_eager = cfg.get("enforce_eager", False)

    engine = LlamaEngine(
        model_name=cfg["model"],
        seed=42,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_model_len=cfg["max_model_len"],
    )

    # Warmup
    engine.generate(
        [[0]*128],
        SamplingParams(temperature=0.0, max_tokens=16),
    )

    results = {}

    for label, input_len, bs, max_tok in cfg["scenarios"]:
        prompts = [
            list(range(input_len))
            for _ in range(bs)
        ]
        sp = SamplingParams(temperature=0.0, max_tokens=max_tok, ignore_eos=True)

        # Warmup
        engine.block_manager.reset()
        engine.generate(prompts, sp)
        torch.cuda.synchronize()
        time.sleep(0.5)

        # Timing
        engine.block_manager.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.generate(prompts, sp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1000
        total_tokens = bs * max_tok
        tok_per_s = total_tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0

        print(f"\n=== kb-nano {label} ===")
        print(f"  Wall time: {wall_ms:.2f} ms")
        print(f"  Tokens: {total_tokens}")
        print(f"  Throughput: {tok_per_s:.0f} tok/s")
        sys.stdout.flush()

        results[label] = {
            "wall_ms": wall_ms,
            "total_tokens": total_tokens,
            "tok_per_s": tok_per_s,
        }

        # CUDA kernel profile
        engine.block_manager.reset()
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            engine.generate(prompts, sp)
            torch.cuda.synchronize()

        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
        print(f"\n=== kb-nano {label} (CUDA kernels) ===")
        print(table)
        sys.stdout.flush()

        prof.export_chrome_trace(f"/tmp/profile_kb_{label}.json")

    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)

    del engine
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
'''


def run_worker(script: str, config: dict, label: str, timeout: int = 1800):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(script)
        script_path = f.name

    output_path = tempfile.mktemp(suffix=".json", dir="/tmp")
    config["output_file"] = output_path

    config_path = tempfile.mktemp(suffix=".json", dir="/tmp")
    with open(config_path, "w") as f:
        json.dump(config, f)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}", flush=True)

    result = subprocess.run(
        [sys.executable, script_path, config_path],
        timeout=timeout,
    )

    os.unlink(script_path)
    os.unlink(config_path)

    data = None
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        os.unlink(output_path)

    if result.returncode != 0:
        print(f"  ERROR: {label} failed (exit {result.returncode})")
        return None

    return data


def main():
    model = "deepseek-ai/DeepSeek-V3.2"
    num_layers = 4
    max_model_len = 2048

    scenarios = [
        ("prefill_512x4", 512, 4, 1),
        ("decode_bs32", 128, 32, 20),
        ("decode_bs128", 128, 128, 20),
    ]

    config = {
        "model": model,
        "num_layers": num_layers,
        "max_model_len": max_model_len,
        "project_root": str(_PROJECT_ROOT),
        "scenarios": scenarios,
        "profile_scenario": "prefill_512x4",
        "profile_dir": "/tmp/vllm_profile",
        "enforce_eager": False,
    }

    print("=" * 70)
    print("  Operator-Level Profiling: kb-nano vs vLLM")
    print(f"  Model: {model}")
    print(f"  Layers: {num_layers}")
    print(f"  enforce_eager: {config['enforce_eager']}")
    print(f"  Scenarios: {[s[0] for s in scenarios]}")
    print("=" * 70, flush=True)

    # Run kb-nano first
    kb_data = run_worker(KB_PROFILER, dict(config), "kb-nano Profiling")

    # Kill any leftover GPU processes
    import signal, time as _time
    my_pid = os.getpid()
    for _ in range(3):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            ).strip()
            if not out:
                break
            for pid_str in out.splitlines():
                pid_str = pid_str.strip()
                if pid_str:
                    try:
                        pid = int(pid_str)
                        if pid != my_pid:
                            os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, ValueError, PermissionError):
                        pass
        except Exception:
            pass
        _time.sleep(5)

    # Run vLLM
    vllm_data = run_worker(VLLM_PROFILER, dict(config), "vLLM Profiling")

    # Print comparison
    print("\n\n" + "=" * 70)
    print("  TIMING COMPARISON")
    print("=" * 70)
    if kb_data and vllm_data:
        print(f"  {'Scenario':<20} {'kb-nano ms':>12} {'vLLM ms':>12} {'ratio':>8}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*8}")
        for label, _, _, _ in scenarios:
            kb_ms = kb_data.get(label, {}).get("wall_ms", 0)
            vllm_ms = vllm_data.get(label, {}).get("wall_ms", 0)
            ratio = kb_ms / vllm_ms if vllm_ms > 0 else float("nan")
            print(f"  {label:<20} {kb_ms:>12.2f} {vllm_ms:>12.2f} {ratio:>8.2f}x")

    print("\n\nChrome traces: /tmp/profile_kb_*.json")
    print("vLLM traces: /tmp/vllm_profile/")


if __name__ == "__main__":
    main()
