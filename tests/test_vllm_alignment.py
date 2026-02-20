#!/usr/bin/env python3
"""
Correctness test: standalone engine vs vLLM.

Runs both engines in separate subprocesses with enforce_eager=True for
deterministic outputs, then compares generated tokens.

Usage:
    python tests/test_vllm_alignment.py --model meta-llama/Llama-3.1-8B-Instruct

    python tests/test_vllm_alignment.py \
        --model meta-llama/Llama-3.1-70B-Instruct mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --tp 4

    python tests/test_vllm_alignment.py --model meta-llama/Llama-3.1-8B-Instruct --max-tokens 200 --seed 123
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

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
# vLLM worker (runs in subprocess)
# ---------------------------------------------------------------------------
VLLM_WORKER = r'''
import json, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cfg = json.loads(sys.argv[1])
    attn_config = AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN)
    llm = LLM(
        model=cfg["model"], seed=cfg["seed"], enforce_eager=True,
        tensor_parallel_size=cfg["tp"],
        attention_config=attn_config,
    )
    sp = SamplingParams(
        temperature=0.0, max_tokens=cfg["max_tokens"], seed=cfg["seed"],
    )

    llm.generate(["warmup"], sp)

    results = []
    for prompt in cfg["prompts"]:
        out = llm.generate([prompt], sp)[0]
        results.append({
            "text": out.outputs[0].text,
            "token_ids": list(out.outputs[0].token_ids),
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": results}, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Standalone worker (runs in subprocess)
# ---------------------------------------------------------------------------
STANDALONE_WORKER = r'''
import json, os, sys
cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["project_root"])

def main():
    cfg = json.loads(sys.argv[1])
    pkg = cfg["package_name"]
    mod = __import__(f"{pkg}.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine = LlamaEngine(
        model_name=cfg["model"], seed=cfg["seed"],
        enforce_eager=True,
        tensor_parallel_size=cfg["tp"],
    )
    sp = SamplingParams(
        temperature=0.0, max_tokens=cfg["max_tokens"], seed=cfg["seed"],
    )

    engine.generate(["warmup"], sp)

    results = []
    for prompt in cfg["prompts"]:
        out = engine.generate([prompt], sp)[0]
        results.append({
            "text": out.generated_text,
            "token_ids": out.token_ids,
        })

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": results}, f)

    del engine

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

    config["output_file"] = output_path

    try:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        result = subprocess.run(
            [sys.executable, script_path, json.dumps(config)],
            timeout=1200,
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_correctness(
    vllm_data: dict, standalone_data: dict, prompts: list[str],
) -> int:
    """Print correctness comparison. Returns number of mismatches."""
    print(f"\n{'=' * 70}")
    print("  CORRECTNESS")
    print(f"{'=' * 70}")

    mismatches = 0
    for i, (vr, sr) in enumerate(
        zip(vllm_data["results"], standalone_data["results"])
    ):
        v_ids = vr["token_ids"]
        s_ids = sr["token_ids"]
        match = v_ids == s_ids

        prompt_preview = prompts[i][:55] + ("..." if len(prompts[i]) > 55 else "")
        ntoks = len(v_ids)

        if match:
            print(f"  #{i} MATCH   ({ntoks:>3} tokens) | {prompt_preview}")
        else:
            mismatches += 1
            min_len = min(len(v_ids), len(s_ids))
            div = next(
                (j for j in range(min_len) if v_ids[j] != s_ids[j]), min_len
            )
            print(f"  #{i} MISMATCH at token {div:>3} | {prompt_preview}")
            print(f"       vLLM : {vr['text'][:70]!r}...")
            print(f"       Ours : {sr['text'][:70]!r}...")

    total = len(prompts)
    print(f"\n  Result: {total - mismatches}/{total} exact matches")
    return mismatches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_model_test(model_name: str, args, project_root: str, package_name: str) -> int:
    """Run correctness test for a single model. Returns number of mismatches."""
    short_name = model_name.split("/")[-1]
    print(f"\n{'#' * 70}")
    print(f"  MODEL: {model_name}")
    print(f"  TP={args.tp}  max_tokens={args.max_tokens}  seed={args.seed}")
    print(f"{'#' * 70}")

    config = {
        "model": model_name,
        "tp": args.tp,
        "seed": args.seed,
        "prompts": PROMPTS,
        "max_tokens": args.max_tokens,
        "project_root": project_root,
        "package_name": package_name,
    }

    vllm_data = run_worker(
        VLLM_WORKER, dict(config),
        f"vLLM  [{short_name}] (TP={args.tp}, eager)",
    )
    standalone_data = run_worker(
        STANDALONE_WORKER, dict(config),
        f"Ours  [{short_name}] (TP={args.tp}, eager)",
    )

    if vllm_data is None or standalone_data is None:
        print(f"\n  ERROR: One or both engines failed for {short_name}.")
        return len(PROMPTS)

    mismatches = report_correctness(vllm_data, standalone_data, PROMPTS)

    if mismatches == 0:
        print(f"\n  PASS [{short_name}]: All outputs are token-identical.")
    else:
        print(f"\n  WARN [{short_name}]: {mismatches}/{len(PROMPTS)} prompts had mismatches.")

    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Correctness test: standalone engine vs vLLM",
    )
    parser.add_argument(
        "--model", nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"],
        help="One or more HuggingFace model names (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100,
        help="Max tokens to generate per prompt (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    models = args.model
    print("=" * 70)
    print("  Standalone vs vLLM — Correctness Test")
    print("=" * 70)
    print(f"  Models     : {', '.join(models)}")
    print(f"  TP         : {args.tp}")
    print(f"  Max tokens : {args.max_tokens}")
    print(f"  Seed       : {args.seed}")
    print(f"  Prompts    : {len(PROMPTS)}")
    print("=" * 70)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(package_dir)
    package_name = os.path.basename(package_dir)

    results = {}
    for model_name in models:
        mismatches = run_model_test(model_name, args, project_root, package_name)
        results[model_name] = mismatches

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    any_fail = False
    for model_name, mismatches in results.items():
        short = model_name.split("/")[-1]
        status = "PASS" if mismatches == 0 else f"WARN ({mismatches}/{len(PROMPTS)} mismatches)"
        if mismatches == len(PROMPTS):
            status = f"FAIL ({mismatches}/{len(PROMPTS)} mismatches)"
            any_fail = True
        print(f"  {short:<45} {status}")
    print("=" * 70)

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
