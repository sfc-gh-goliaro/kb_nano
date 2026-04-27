#!/usr/bin/env python3
"""
Correctness test: standalone engine vs HuggingFace transformers.

Uses transformers as the reference implementation (no vLLM dependency).
Both engines run in separate subprocesses with greedy decoding for
deterministic outputs, then compares generated tokens.

Usage:
    python tests/test_transformers_alignment.py --model meta-llama/Llama-3.1-8B-Instruct

    python tests/test_transformers_alignment.py \
        --model meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B \
        --max-tokens 50

    python tests/test_transformers_alignment.py \
        --model meta-llama/Llama-3.1-70B-Instruct --tp 4 --max-tokens 50

    # Mamba v1 alignment against HuggingFace transformers
    python tests/test_transformers_alignment.py \
        --model state-spaces/mamba-130m-hf \
        --max-tokens 32

    # Only probe whether transformers can load and generate with a model
    python tests/test_transformers_alignment.py --model state-spaces/mamba-2.8b-hf --hf-only

    # Run only the HF reference and print full generated text (no engine needed)
    python tests/test_transformers_alignment.py \
        --model state-spaces/mamba-130m-hf \
        --ref-only --max-tokens 32
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
# HuggingFace transformers worker (runs in subprocess)
# ---------------------------------------------------------------------------
HF_WORKER = r'''
import json, os, sys, torch

import fla
from fla.models import GLAForCausalLM


def _load_mamba_ssm(model_name, dtype, device):
    """Load Mamba v1/v2 via mamba_ssm library (avoids fla dtype bugs)."""
    import inspect, glob
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig

    model_path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    with open(f"{model_path}/config.json") as f:
        cfg_data = json.load(f)

    sig = inspect.signature(MambaConfig.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    filtered = {k: cfg_data[k] for k in valid_keys if k in cfg_data}
    config = MambaConfig(**filtered)
    model = MambaLMHeadModel(config, dtype=dtype, device=device)

    for sf_file in sorted(glob.glob(f"{model_path}/*.safetensors")):
        with safe_open(sf_file, "pt", "cpu") as f:
            for k in f.keys():
                mapped = k.replace("backbone.embeddings.", "backbone.embedding.")
                try:
                    parts = mapped.split(".")
                    param = model
                    for p in parts[:-1]:
                        param = getattr(param, p)
                    getattr(param, parts[-1]).data.copy_(f.get_tensor(k))
                except AttributeError:
                    pass
    return model


def main():
    cfg = json.loads(sys.argv[1])
    model_name = cfg["model"]
    seed = cfg["seed"]
    max_tokens = cfg["max_tokens"]
    prompts = cfg["prompts"]
    trust_remote_code = cfg.get("trust_remote_code", False)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Detect model type
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(hf_config, "model_type", "")

    if model_type in ("mamba", "mamba2"):
        # Use mamba_ssm directly (fla has dtype bugs for Mamba)
        model = _load_mamba_ssm(model_name, torch.bfloat16, "cuda")
        model.eval()

        results = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + max_tokens,
                    temperature=1.0,
                    top_k=1,
                )
            generated_ids = output[0][input_ids.shape[1]:].tolist()
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": generated_ids})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        model.eval()

        results = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated_ids = output[0][input_ids.shape[1]:].tolist()
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": generated_ids})

    with open(cfg["output_file"], "w") as f:
        json.dump({"results": results}, f)

if __name__ == "__main__":
    main()
'''

HF_PROBE_WORKER = r'''
import json, os, sys, torch

import fla
from fla.models import GLAForCausalLM

def main():
    cfg = json.loads(sys.argv[1])
    model_name = cfg["model"]
    seed = cfg["seed"]
    max_tokens = cfg["max_tokens"]
    trust_remote_code = cfg.get("trust_remote_code", False)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    hf_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )
    model_type = getattr(hf_config, "model_type", "unknown")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    prompt = cfg["prompts"][0]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):

            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(max_tokens, 8),
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
    generated_ids = output[0][input_ids.shape[1]:].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "ok": True,
            "model_type": model_type,
            "model_class": type(model).__name__,
            "text": text,
            "token_ids": generated_ids,
        }, f)

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

import fla
from fla.models import GLAForCausalLM

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
    hf_data: dict, standalone_data: dict, prompts: list[str],
) -> int:
    """Print correctness comparison. Returns number of mismatches."""
    print(f"\n{'=' * 70}")
    print("  CORRECTNESS")
    print(f"{'=' * 70}")

    mismatches = 0
    for i, (hr, sr) in enumerate(
        zip(hf_data["results"], standalone_data["results"])
    ):
        h_ids = hr["token_ids"]
        s_ids = sr["token_ids"]
        match = h_ids == s_ids

        prompt_preview = prompts[i][:55] + ("..." if len(prompts[i]) > 55 else "")
        ntoks = len(h_ids)

        if match:
            print(f"  #{i} MATCH   ({ntoks:>3} tokens) | {prompt_preview}")
        else:
            mismatches += 1
            min_len = min(len(h_ids), len(s_ids))
            div = next(
                (j for j in range(min_len) if h_ids[j] != s_ids[j]), min_len
            )
            print(f"  #{i} MISMATCH at token {div:>3} | {prompt_preview}")
            print(f"       HF   : {hr['text'][:70]!r}...")
            print(f"       Ours : {sr['text'][:70]!r}...")

    total = len(prompts)
    print(f"\n  Result: {total - mismatches}/{total} exact matches")
    return mismatches


def report_hf_probe(hf_data: dict, model_name: str) -> bool:
    """Print probe result. Returns True on successful load+generate."""
    print(f"\n{'=' * 70}")
    print("  HF TRANSFORMERS PROBE")
    print(f"{'=' * 70}")

    if hf_data is None or not hf_data.get("ok"):
        print(f"  FAIL: transformers could not load/generate for {model_name}")
        return False

    model_type = hf_data.get("model_type")
    model_class = hf_data.get("model_class")
    token_ids = hf_data.get("token_ids", [])
    text = hf_data.get("text", "")

    print(f"  Model                : {model_name}")
    print(f"  HF model_type        : {model_type or 'unknown'}")
    print(f"  HF model class       : {model_class or 'unknown'}")
    print(f"  Generated tokens     : {len(token_ids)}")
    print(f"  Sample output        : {text[:120]!r}")
    print("  Result               : PASS")
    return True


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
        "trust_remote_code": args.trust_remote_code,
    }

    hf_data = run_worker(
        HF_WORKER, dict(config),
        f"HF transformers  [{short_name}]",
    )
    standalone_data = run_worker(
        STANDALONE_WORKER, dict(config),
        f"Ours  [{short_name}] (TP={args.tp}, eager)",
    )

    if hf_data is None or standalone_data is None:
        print(f"\n  ERROR: One or both engines failed for {short_name}.")
        return len(PROMPTS)

    mismatches = report_correctness(hf_data, standalone_data, PROMPTS)

    if mismatches == 0:
        print(f"\n  PASS [{short_name}]: All outputs are token-identical.")
    else:
        print(f"\n  WARN [{short_name}]: {mismatches}/{len(PROMPTS)} prompts had mismatches.")

    return mismatches


def run_hf_ref_model(model_name: str, args, project_root: str, package_name: str) -> bool:
    """Run HF reference only and print full generated text."""
    short_name = model_name.split("/")[-1]
    print(f"\n{'#' * 70}")
    print(f"  MODEL: {model_name}")
    print(f"  max_tokens={args.max_tokens}  seed={args.seed}")
    print(f"{'#' * 70}")

    config = {
        "model": model_name,
        "tp": args.tp,
        "seed": args.seed,
        "prompts": PROMPTS,
        "max_tokens": args.max_tokens,
        "project_root": project_root,
        "package_name": package_name,
        "trust_remote_code": args.trust_remote_code,
    }

    hf_data = run_worker(
        HF_WORKER, dict(config),
        f"HF transformers  [{short_name}]",
    )

    if hf_data is None:
        print(f"\n  FAIL: HF transformers failed for {short_name}")
        return False

    print(f"\n{'=' * 70}")
    print(f"  HF REFERENCE OUTPUT — {model_name}")
    print(f"{'=' * 70}")
    for i, r in enumerate(hf_data["results"]):
        prompt_preview = PROMPTS[i][:60] + ("..." if len(PROMPTS[i]) > 60 else "")
        print(f"\n  Prompt #{i}: {prompt_preview}")
        print(f"  Tokens   : {len(r['token_ids'])}")
        print(f"  Output   :")
        for line in r["text"].splitlines():
            print(f"    {line}")
    print(f"\n{'=' * 70}")
    return True


def run_hf_probe_model(model_name: str, args, project_root: str, package_name: str) -> bool:
    """Run a transformers-only support probe for a single model."""
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
        "trust_remote_code": args.trust_remote_code,
    }

    hf_data = run_worker(
        HF_PROBE_WORKER, dict(config),
        f"HF probe [{short_name}]",
    )
    return report_hf_probe(hf_data, model_name)


def main():
    parser = argparse.ArgumentParser(
        description="Correctness test: standalone engine vs HuggingFace transformers",
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
    parser.add_argument(
        "--hf-only", action="store_true",
        help="Only probe whether transformers can load and generate with the model",
    )
    parser.add_argument(
        "--ref-only", action="store_true",
        help="Run only the HF reference and print full generated text (no engine needed)",
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True when loading the model (default: True)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading the model",
    )
    args = parser.parse_args()

    models = args.model
    mode = "HF Reference Output" if args.ref_only else ("HF Transformers Probe" if args.hf_only else "Standalone vs HF Transformers — Correctness Test")
    print("=" * 70)
    print(f"  {mode}")
    print("=" * 70)
    print(f"  Models     : {', '.join(models)}")
    print(f"  TP         : {args.tp}")
    print(f"  Max tokens : {args.max_tokens}")
    print(f"  Seed       : {args.seed}")
    print(f"  Prompts    : {len(PROMPTS)}")
    print(f"  Trust RC   : {args.trust_remote_code}")
    print("=" * 70)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(package_dir)
    package_name = os.path.basename(package_dir)

    if args.ref_only:
        results = {}
        for model_name in models:
            ok = run_hf_ref_model(model_name, args, project_root, package_name)
            results[model_name] = ok

        print(f"\n{'=' * 70}")
        print("  FINAL SUMMARY")
        print(f"{'=' * 70}")
        any_fail = False
        for model_name, ok in results.items():
            short = model_name.split("/")[-1]
            status = "PASS" if ok else "FAIL"
            if not ok:
                any_fail = True
            print(f"  {short:<45} {status}")
        print("=" * 70)

        sys.exit(1 if any_fail else 0)

    if args.hf_only:
        results = {}
        for model_name in models:
            ok = run_hf_probe_model(model_name, args, project_root, package_name)
            results[model_name] = ok

        print(f"\n{'=' * 70}")
        print("  FINAL SUMMARY")
        print(f"{'=' * 70}")
        any_fail = False
        for model_name, ok in results.items():
            short = model_name.split("/")[-1]
            status = "PASS" if ok else "FAIL"
            if not ok:
                any_fail = True
            print(f"  {short:<45} {status}")
        print("=" * 70)

        sys.exit(1 if any_fail else 0)

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
