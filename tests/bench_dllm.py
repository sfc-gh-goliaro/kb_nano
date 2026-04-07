#!/usr/bin/env python3
"""Benchmark repo-native LLaDA against official HF and Fast-dLLM baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import subprocess

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PACKAGE_DIR))

from bench.utils.worker import run_worker


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


PROMPT = "Write a concise explanation of why regularization helps neural networks generalize."


DLLM_WORKER = r'''
import json, os, sys, time
import importlib.util
from pathlib import Path
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

pkg_root = Path(cfg["project_root"])
spec = importlib.util.spec_from_file_location(
    "kb_nano", pkg_root / "__init__.py",
    submodule_search_locations=[str(pkg_root)],
)
kb_nano = importlib.util.module_from_spec(spec)
sys.modules["kb_nano"] = kb_nano
spec.loader.exec_module(kb_nano)

from transformers import AutoModelForCausalLM, AutoTokenizer

from kb_nano.infra.dllm_engine import (
    LLaDAEngine,
    DLLMSamplingParams,
    masked_diffusion_generate,
    masked_diffusion_generate_with_dual_cache,
    masked_diffusion_generate_with_prefix_cache,
)


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    device = cfg.get("device", "cuda")
    dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float16
    model_name = cfg["model"]
    prompt = cfg["prompt"]
    num_prompts = cfg["num_prompts"]
    prompt_batch = [prompt] * num_prompts

    backend = cfg["backend"]

    if backend == "ours":
        engine = LLaDAEngine(model_name=model_name, dtype=dtype, device=device)
        tokenizer = engine.tokenizer
        sp = DLLMSamplingParams(
            gen_length=cfg["gen_length"],
            steps=cfg["steps"],
            block_length=cfg["block_length"],
            temperature=cfg["temperature"],
            remasking=cfg["remasking"],
            threshold=cfg.get("threshold"),
            decode_mode=cfg.get("ours_backend", "vanilla"),
        )
        encoded = engine._encode_prompt(prompt)
        prompt_tensor = torch.tensor(encoded, dtype=torch.long, device=engine.model.device).unsqueeze(0)
        logits = engine.model(prompt_tensor).logits

        for _ in range(cfg["warmup_iters"]):
            engine.generate(prompt_batch, sp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = None
        for _ in range(cfg["measure_iters"]):
            outputs = engine.generate(prompt_batch, sp)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        result = {
            "backend": "ours",
            "decode_mode": sp.decode_mode,
            "elapsed": elapsed / cfg["measure_iters"],
            "tokens_per_second": (num_prompts * cfg["gen_length"]) / (elapsed / cfg["measure_iters"]),
            "nfe": outputs[0].nfe,
            "prompt_ids": encoded,
            "logits_shape": list(logits.shape),
            "logits_sample": logits[:, -1, :2048].float().cpu().tolist(),
            "outputs": [{"token_ids": o.token_ids, "text": o.generated_text} for o in outputs],
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = tokenizer(prompt_text)["input_ids"]
        prompt_tensor = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

        if backend == "hf":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(device).eval()
            logits = model(prompt_tensor).logits
            run_generate = masked_diffusion_generate
        else:
            fastdllm_root = cfg.get("fastdllm_root")
            if not fastdllm_root:
                raise ValueError("fastdllm_root is required for Fast-dLLM reference backends")
            fastdllm_llada = str(Path(fastdllm_root) / "llada")
            if fastdllm_llada not in sys.path:
                sys.path.insert(0, fastdllm_llada)
            from generate import generate_with_dual_cache, generate_with_prefix_cache
            from model.modeling_llada import LLaDAModelLM as FastDLLMLLaDAModelLM

            model = FastDLLMLLaDAModelLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(device).eval()
            logits = model(prompt_tensor).logits
            if backend == "fastdllm-prefix":
                run_generate = generate_with_prefix_cache
            elif backend == "fastdllm-dual":
                run_generate = generate_with_dual_cache
            else:
                raise ValueError(f"Unknown backend: {backend}")

        for _ in range(cfg["warmup_iters"]):
            _, nfe = run_generate(
                model,
                prompt_tensor,
                steps=cfg["steps"],
                gen_length=cfg["gen_length"],
                block_length=cfg["block_length"],
                temperature=cfg["temperature"],
                remasking=cfg["remasking"],
                mask_id=cfg["mask_token_id"],
                threshold=cfg.get("threshold"),
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = None
        nfe = None
        for _ in range(cfg["measure_iters"]):
            outputs, nfe = run_generate(
                model,
                prompt_tensor.expand(num_prompts, -1),
                steps=cfg["steps"],
                gen_length=cfg["gen_length"],
                block_length=cfg["block_length"],
                temperature=cfg["temperature"],
                remasking=cfg["remasking"],
                mask_id=cfg["mask_token_id"],
                threshold=cfg.get("threshold"),
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        result = {
            "backend": backend,
            "elapsed": elapsed / cfg["measure_iters"],
            "tokens_per_second": (num_prompts * cfg["gen_length"]) / (elapsed / cfg["measure_iters"]),
            "nfe": nfe,
            "prompt_ids": encoded,
            "logits_shape": list(logits.shape),
            "logits_sample": logits[:, -1, :2048].float().cpu().tolist(),
            "outputs": [
                {
                    "token_ids": seq[len(encoded):].tolist(),
                    "text": tokenizer.decode(seq[len(encoded):], skip_special_tokens=True),
                }
                for seq in outputs
            ],
        }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())
    os._exit(0)


if __name__ == "__main__":
    main()
'''


def _cosine(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)


def _mae(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def _compare_outputs(ours, ref):
    token_matches = []
    seq_exact = []
    for o, r in zip(ours["outputs"], ref["outputs"]):
        ot, rt = o["token_ids"], r["token_ids"]
        common_len = min(len(ot), len(rt))
        match = sum(int(a == b) for a, b in zip(ot[:common_len], rt[:common_len]))
        token_matches.append(match / max(common_len, 1))
        seq_exact.append(int(ot == rt))
    return {
        "throughput_ratio": ours["tokens_per_second"] / ref["tokens_per_second"],
        "logits_cosine": _cosine(ours["logits_sample"], ref["logits_sample"]),
        "logits_mae": _mae(ours["logits_sample"], ref["logits_sample"]),
        "token_match_rate": sum(token_matches) / len(token_matches),
        "sequence_exact_match_rate": sum(seq_exact) / len(seq_exact),
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark repo-native LLaDA vs official HF / Fast-dLLM baselines")
    ap.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--prompt", type=str, default=PROMPT)
    ap.add_argument("--num-prompts", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--remasking", type=str, default="low_confidence")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--warmup-iters", type=int, default=1)
    ap.add_argument("--measure-iters", type=int, default=3)
    ap.add_argument(
        "--ours-backend",
        type=str,
        default="vanilla",
        choices=["vanilla", "prefix", "dual"],
        help="Decode backend for our implementation.",
    )
    ap.add_argument(
        "--reference-backends",
        type=str,
        default="hf",
        help="Comma-separated baseline backends: hf, fastdllm-prefix, fastdllm-dual",
    )
    ap.add_argument(
        "--fastdllm-root",
        type=str,
        default="third_party/Fast-dLLM",
        help="Path to the official Fast-dLLM repo when using fastdllm-* baselines.",
    )
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    gpu = _detect_gpu_name()
    out_dir = Path(args.output_dir or f"tests/results/{gpu}/llada-8b-instruct")
    out_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "project_root": str(_PACKAGE_DIR),
        "model": args.model,
        "prompt": args.prompt,
        "num_prompts": args.num_prompts,
        "gen_length": args.gen_length,
        "steps": args.steps,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "remasking": args.remasking,
        "threshold": args.threshold,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "mask_token_id": 126336,
        "use_bf16": True,
        "device": "cuda",
        "ours_backend": args.ours_backend,
    }

    ours = run_worker(DLLM_WORKER, {**common, "backend": "ours"}, "kb-nano LLaDA")
    references = {}
    comparisons = {}
    if not args.skip_reference:
        backends = [b.strip() for b in args.reference_backends.split(",") if b.strip()]
        for backend in backends:
            if backend.startswith("fastdllm") and not Path(args.fastdllm_root).exists():
                raise FileNotFoundError(
                    f"Fast-dLLM baseline requested but repo not found: {args.fastdllm_root}"
                )
            label = {
                "hf": "official HF LLaDA",
                "fastdllm-prefix": "official Fast-dLLM (prefix cache)",
                "fastdllm-dual": "official Fast-dLLM (dual cache)",
            }.get(backend, backend)
            ref = run_worker(
                DLLM_WORKER,
                {**common, "backend": backend, "fastdllm_root": args.fastdllm_root},
                label,
            )
            references[backend] = ref
            if ours and ref:
                comparisons[backend] = _compare_outputs(ours, ref)

    results = {"model": args.model, "ours": ours, "references": references, "comparisons": comparisons}

    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
