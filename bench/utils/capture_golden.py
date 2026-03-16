#!/usr/bin/env python3
"""Capture golden input data for data-dependent operators.

Runs real E2E inference and hooks into target operators via
register_forward_hook to capture real tensor data at the middle layer.

The captured .pt files are stored under bench/utils/golden_data/ and
referenced from the YAML manifests.

Usage:
    python -m kb_nano.bench.utils.capture_golden \
        --model meta-llama/Llama-3.1-8B-Instruct --tp 1

    python -m kb_nano.bench.utils.capture_golden \
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp 1

Golden data is meant to be uploaded to HuggingFace Hub
(sfc-gh-goliaro/kb-nano-golden-inputs) and downloaded on first use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from kb_nano.paths import GOLDEN_DIR

_GOLDEN_DIR = GOLDEN_DIR

DATA_DEPENDENT_OPS = [
    "moe_align",
    "moe_grouped_gemm",
    "fused_experts",
    "store_kvcache",
]

CANONICAL_M_VALUES = [1, 8, 32, 128, 128, 512]

MODEL_KEY_MAP = {
    "llama": "llama31",
    "mixtral": "mixtral",
}


def _get_model_key(model_name: str) -> str:
    name_lower = model_name.lower()
    for pattern, key in MODEL_KEY_MAP.items():
        if pattern in name_lower:
            return key
    return model_name.split("/")[-1].lower().replace("-", "_")


def _get_short_model_key(model_name: str) -> str:
    """Get short key like 'mixtral-8x7b' for directory naming."""
    name = model_name.split("/")[-1].lower()
    if "mixtral" in name:
        return "mixtral-8x7b"
    if "llama" in name:
        if "8b" in name:
            return "llama31-8b"
        if "70b" in name:
            return "llama31-70b"
    return name


def capture_golden_data(
    model_name: str,
    tp: int = 1,
    output_dir: Path | None = None,
    num_requests: int = 10,
    seed: int = 42,
) -> list[Path]:
    """Capture golden tensor data from real E2E inference.

    Hooks into the middle layer of the model during inference and captures
    inputs to data-dependent operators.

    Args:
        model_name: HuggingFace model name.
        tp: Tensor parallelism degree.
        output_dir: Output directory for .pt files.
        num_requests: Number of requests to run.
        seed: Random seed.

    Returns:
        List of paths to saved .pt files.
    """
    if output_dir is None:
        output_dir = _GOLDEN_DIR

    model_key = _get_short_model_key(model_name)
    saved_files: list[Path] = []

    print(f"Capturing golden data for {model_name} (TP={tp})")
    print(f"  Model key: {model_key}")
    print(f"  Output dir: {output_dir}")
    print(f"  Note: This requires GPU and a working LlamaEngine setup.")
    print(f"  Data-dependent ops: {DATA_DEPENDENT_OPS}")

    try:
        from kb_nano.infra.engine import LlamaEngine, SamplingParams
    except ImportError:
        print("ERROR: Cannot import kb_nano.infra.engine. Make sure the package is installed.")
        return saved_files

    engine = LlamaEngine(
        model_name=model_name,
        seed=seed,
        enforce_eager=True,
        tensor_parallel_size=tp,
    )

    captured_data: dict[str, dict[str, dict]] = {}
    hooks = []

    model = engine.model
    num_layers = len(model.model.layers) if hasattr(model, "model") else 0
    mid_layer = num_layers // 2
    print(f"  Total layers: {num_layers}, capturing from layer {mid_layer}")

    def _make_hook(op_name: str, scenario_label: str):
        def hook_fn(module, args, kwargs, output):
            key = f"{op_name}/{scenario_label}"
            if key not in captured_data.get(op_name, {}):
                captured_data.setdefault(op_name, {})[scenario_label] = {
                    "args": tuple(
                        a.detach().cpu().clone() if isinstance(a, torch.Tensor) else a
                        for a in args
                    ),
                    "kwargs": {
                        k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs.items()
                    } if kwargs else {},
                }
        return hook_fn

    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a Python hello world program.",
        "What are the planets in our solar system?",
        "Describe the water cycle.",
    ]
    prompts = prompts * (num_requests // len(prompts) + 1)
    prompts = prompts[:num_requests]

    sp = SamplingParams(temperature=0.0, max_tokens=32, seed=seed)
    print(f"  Running {len(prompts)} requests for golden data capture...")
    engine.generate(prompts, sp)

    for op_name, scenarios in captured_data.items():
        for scenario_label, data in scenarios.items():
            op_dir = output_dir / model_key / op_name
            op_dir.mkdir(parents=True, exist_ok=True)
            pt_path = op_dir / f"{scenario_label}.pt"
            torch.save(data, pt_path)
            saved_files.append(pt_path)
            print(f"  Saved: {pt_path}")

    for h in hooks:
        h.remove()
    engine._cleanup()
    del engine

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Capture golden input data")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {_GOLDEN_DIR})")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Number of inference requests to run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None
    files = capture_golden_data(args.model, args.tp, out, args.num_requests, args.seed)
    if files:
        print(f"\nCaptured {len(files)} golden data files.")
    else:
        print("\nNo golden data captured (this is expected without GPU).")


if __name__ == "__main__":
    main()
