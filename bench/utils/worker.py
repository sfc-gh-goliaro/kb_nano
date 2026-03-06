"""Subprocess worker runner for benchmark isolation.

Runs benchmark workers in clean subprocesses to avoid import contamination
and ensure CUDA graphs / torch.compile operate in a pristine environment.

Refactored from tests/bench_throughput.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


def run_worker(script: str, config: dict, label: str, timeout: int = 3600) -> dict | None:
    """Run a worker script in a subprocess and return parsed JSON output.

    Args:
        script: Python source code to execute.
        config: JSON-serializable configuration dict passed as argv[1].
                An ``output_file`` key is added automatically.
        label: Human-readable label printed before/after execution.
        timeout: Maximum wall-clock seconds before the subprocess is killed.

    Returns:
        Parsed JSON dict written by the worker to ``output_file``, or None on
        failure.
    """
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
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  ERROR: {label} failed with exit code {result.returncode}")
            return None

        with open(output_path) as f:
            return json.loads(f.read())
    except subprocess.TimeoutExpired:
        print(f"  ERROR: {label} timed out after {timeout}s")
        return None
    finally:
        os.unlink(script_path)
        os.unlink(config_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


KB_NANO_WORKER = r'''
import json, sys, time
with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    pkg = cfg["package_name"]

    if not cfg.get("no_candidate_kernels", False):
        swapper = __import__(
            f"{pkg}.infra.kernel_swapper",
            fromlist=["discover_candidates", "apply_candidates", "print_candidate_summary"],
        )
        candidates = swapper.discover_candidates()
        if candidates:
            swapper.print_candidate_summary(candidates)
            swapper.apply_candidates(candidates)

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

    prompts = cfg["prompts"]
    temperature = cfg.get("temperature", 1.0)
    top_p = cfg.get("top_p", 1.0)
    output_lens = cfg["output_lens"]
    ignore_eos = cfg.get("ignore_eos", True)

    sp_list = [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=ol,
            ignore_eos=ignore_eos,
        )
        for ol in output_lens
    ]

    # Warmup
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, sp_list)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_input_tokens = sum(
        len(engine.tokenizer.encode(p)) if isinstance(p, str) else len(p)
        for p in prompts
    )
    total_output_tokens = sum(len(o.token_ids) for o in outputs)

    result = {
        "elapsed": elapsed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    if cfg.get("save_outputs", False):
        result["outputs"] = [
            {
                "prompt": o.prompt,
                "generated_text": o.generated_text,
                "token_ids": o.token_ids,
            }
            for o in outputs
        ]

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

    del engine

if __name__ == "__main__":
    main()
'''
