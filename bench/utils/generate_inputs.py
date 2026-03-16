#!/usr/bin/env python3
"""Generate YAML input manifests for the kernel Input Registry.

Derives operator input shapes from HuggingFace model configs with TP-aware
shape folding and deduplication. Outputs YAML files under bench/utils/inputs/.

Usage:
    python -m kb_nano.bench.utils.generate_inputs
    python -m kb_nano.bench.utils.generate_inputs --models meta-llama/Llama-3.1-8B-Instruct
    python -m kb_nano.bench.utils.generate_inputs --output-dir bench/utils/inputs/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from kb_nano.paths import INPUTS_DIR

_INPUTS_DIR = INPUTS_DIR

CANONICAL_M_VALUES = {
    "decode": [1, 8, 32, 128],
    "prefill": [128, 512],
}

DEFAULT_MODELS = {
    "llama31-8b": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 32,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000,
        "max_position_embeddings": 131072,
    },
    "llama31-70b": {
        "hf_name": "meta-llama/Llama-3.1-70B-Instruct",
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 80,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "max_position_embeddings": 131072,
    },
    "mixtral-8x7b": {
        "hf_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000,
        "max_position_embeddings": 32768,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe": True,
    },
}

APPLICABLE_TP_DEGREES = {
    "llama31-8b": [1],
    "llama31-70b": [4],
    "mixtral-8x7b": [1, 2],
}

COMMUNICATION_OPS = {"allreduce", "parallel_linear", "parallel_embedding"}


def _tp_adjusted(value: int, tp: int) -> int:
    return value // tp


def _generate_rms_norm(model_key: str, cfg: dict, tp: int) -> list[dict]:
    h = _tp_adjusted(cfg["hidden_size"], 1)
    eps = cfg["rms_norm_eps"]
    scenarios = []
    for m in CANONICAL_M_VALUES["decode"]:
        scenarios.append({
            "name": f"{model_key}/decode-bs{m}/tp{tp}",
            "init_args": {"hidden_size": h, "eps": eps},
            "inputs": {
                "x": {"shape": [m, h], "dtype": "bfloat16"},
                "residual": {"shape": [m, h], "dtype": "bfloat16"},
            },
        })
    for m in CANONICAL_M_VALUES["prefill"]:
        scenarios.append({
            "name": f"{model_key}/prefill-{m}/tp{tp}",
            "init_args": {"hidden_size": h, "eps": eps},
            "inputs": {
                "x": {"shape": [m, h], "dtype": "bfloat16"},
                "residual": None,
            },
        })
    return scenarios


def _generate_silu_and_mul(model_key: str, cfg: dict, tp: int) -> list[dict]:
    gate_up = _tp_adjusted(cfg["intermediate_size"], tp) * 2
    scenarios = []
    for m in CANONICAL_M_VALUES["decode"]:
        scenarios.append({
            "name": f"{model_key}/decode-bs{m}/tp{tp}",
            "init_args": {},
            "inputs": {"x": {"shape": [m, gate_up], "dtype": "bfloat16"}},
        })
    for m in CANONICAL_M_VALUES["prefill"]:
        scenarios.append({
            "name": f"{model_key}/prefill-{m}/tp{tp}",
            "init_args": {},
            "inputs": {"x": {"shape": [m, gate_up], "dtype": "bfloat16"}},
        })
    return scenarios


def _generate_rotary_emb(model_key: str, cfg: dict, tp: int) -> list[dict]:
    head_dim = cfg["head_dim"]
    nq = _tp_adjusted(cfg["num_attention_heads"], tp)
    nkv = _tp_adjusted(cfg["num_key_value_heads"], tp)
    init = {
        "head_size": head_dim,
        "rotary_dim": head_dim,
        "max_position_embeddings": cfg["max_position_embeddings"],
        "base": cfg["rope_theta"],
        "is_neox_style": True,
    }
    scenarios = []
    for m in CANONICAL_M_VALUES["decode"]:
        scenarios.append({
            "name": f"{model_key}/decode-bs{m}/tp{tp}",
            "init_args": dict(init),
            "inputs": {
                "positions": {"shape": [m], "dtype": "int64"},
                "query": {"shape": [m, nq, head_dim], "dtype": "bfloat16"},
                "key": {"shape": [m, nkv, head_dim], "dtype": "bfloat16"},
            },
        })
    for m in CANONICAL_M_VALUES["prefill"]:
        scenarios.append({
            "name": f"{model_key}/prefill-{m}/tp{tp}",
            "init_args": dict(init),
            "inputs": {
                "positions": {"shape": [m], "dtype": "int64"},
                "query": {"shape": [m, nq, head_dim], "dtype": "bfloat16"},
                "key": {"shape": [m, nkv, head_dim], "dtype": "bfloat16"},
            },
        })
    return scenarios


def _generate_linear(model_key: str, cfg: dict, tp: int) -> list[dict]:
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    nq = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]

    proj_sizes = {
        "q-proj": (h, _tp_adjusted(nq * head_dim, tp)),
        "k-proj": (h, _tp_adjusted(nkv * head_dim, tp)),
        "v-proj": (h, _tp_adjusted(nkv * head_dim, tp)),
        "o-proj": (_tp_adjusted(nq * head_dim, tp), h),
        "gate-proj": (h, _tp_adjusted(inter, tp)),
        "up-proj": (h, _tp_adjusted(inter, tp)),
        "down-proj": (_tp_adjusted(inter, tp), h),
    }

    seen_shapes: set[tuple] = set()
    scenarios = []
    for proj_name, (in_size, out_size) in proj_sizes.items():
        for m in [1, 32, 512]:
            shape_key = (m, in_size, out_size)
            if shape_key in seen_shapes:
                continue
            seen_shapes.add(shape_key)
            phase = "decode" if m <= 128 else "prefill"
            label = f"decode-bs{m}" if phase == "decode" else f"prefill-{m}"
            scenarios.append({
                "name": f"{model_key}/{proj_name}/{label}/tp{tp}",
                "init_args": {"input_size": in_size, "output_size": out_size, "bias": False},
                "inputs": {"x": {"shape": [m, in_size], "dtype": "bfloat16"}},
            })
    return scenarios


def _generate_embedding(model_key: str, cfg: dict, tp: int) -> list[dict]:
    scenarios = []
    for m in [1, 32, 512]:
        label = f"decode-bs{m}" if m <= 128 else f"prefill-{m}"
        scenarios.append({
            "name": f"{model_key}/{label}/tp{tp}",
            "init_args": {"num_embeddings": cfg["vocab_size"], "embedding_dim": cfg["hidden_size"]},
            "inputs": {"input_ids": {"shape": [m], "dtype": "int64"}},
        })
    return scenarios


def _generate_moe_ops(model_key: str, cfg: dict, tp: int) -> dict[str, list[dict]]:
    """Generate scenarios for MoE-specific operators (data-dependent)."""
    if not cfg.get("moe"):
        return {}
    h = cfg["hidden_size"]
    ne = cfg["num_experts"]
    topk = cfg["num_experts_per_tok"]
    result: dict[str, list[dict]] = {}

    align_scenarios = []
    for m in CANONICAL_M_VALUES["decode"] + CANONICAL_M_VALUES["prefill"]:
        phase = "decode" if m <= 128 else "prefill"
        label = f"decode-bs{m}" if phase == "decode" else f"prefill-{m}"
        align_scenarios.append({
            "name": f"{model_key}/{label}/tp{tp}",
            "init_args": {},
            "inputs": {
                "topk_ids": {"shape": [m, topk], "dtype": "int32"},
                "block_size": 128,
                "num_experts": ne,
            },
            "golden": f"{model_key}/moe_align/{label}-tp{tp}.pt",
        })
    result["moe_align"] = align_scenarios

    gemm_scenarios = []
    for m in [1, 32, 512]:
        phase = "decode" if m <= 128 else "prefill"
        label = f"decode-bs{m}" if phase == "decode" else f"prefill-{m}"
        gemm_scenarios.append({
            "name": f"{model_key}/{label}/tp{tp}",
            "init_args": {},
            "inputs": {
                "hidden_states": {"shape": [m, h], "dtype": "bfloat16"},
                "topk_ids": {"shape": [m, topk], "dtype": "int32"},
                "topk_weights": {"shape": [m, topk], "dtype": "float32"},
            },
            "golden": f"{model_key}/moe_grouped_gemm/{label}-tp{tp}.pt",
        })
    result["moe_grouped_gemm"] = gemm_scenarios

    sum_scenarios = []
    for m in [1, 32, 512]:
        phase = "decode" if m <= 128 else "prefill"
        label = f"decode-bs{m}" if phase == "decode" else f"prefill-{m}"
        sum_scenarios.append({
            "name": f"{model_key}/{label}/tp{tp}",
            "init_args": {},
            "inputs": {"x": {"shape": [topk, m, h], "dtype": "bfloat16"}},
        })
    result["moe_sum"] = sum_scenarios

    return result


def _deduplicate_scenarios(scenarios: list[dict]) -> list[dict]:
    """Remove scenarios with identical input shapes, keeping the first occurrence."""
    seen: set[str] = set()
    deduped = []
    for s in scenarios:
        input_key = str(s.get("inputs", {}))
        if input_key not in seen:
            seen.add(input_key)
            deduped.append(s)
    return deduped


def generate_all(output_dir: Path | None = None) -> dict[str, dict]:
    if output_dir is None:
        output_dir = _INPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    operators: dict[str, list[dict]] = defaultdict(list)

    for model_key, cfg in DEFAULT_MODELS.items():
        tp_degrees = APPLICABLE_TP_DEGREES.get(model_key, [1])
        for tp in tp_degrees:
            operators["rms_norm"].extend(_generate_rms_norm(model_key, cfg, tp))
            operators["silu_and_mul"].extend(_generate_silu_and_mul(model_key, cfg, tp))
            operators["rotary_emb"].extend(_generate_rotary_emb(model_key, cfg, tp))
            operators["linear"].extend(_generate_linear(model_key, cfg, tp))
            operators["embedding"].extend(_generate_embedding(model_key, cfg, tp))
            moe_ops = _generate_moe_ops(model_key, cfg, tp)
            for op_name, scenarios in moe_ops.items():
                operators[op_name].extend(scenarios)

    for op_name in operators:
        operators[op_name] = _deduplicate_scenarios(operators[op_name])

    yaml_data: dict[str, dict] = {}
    for op_name in sorted(operators):
        yaml_data[op_name] = {"scenarios": operators[op_name]}

    output_file = output_dir / "llm.yaml"
    with open(output_file, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print(f"Generated {output_file} with {len(operators)} operators, "
          f"{sum(len(v) for v in operators.values())} total scenarios")

    return yaml_data


def main():
    parser = argparse.ArgumentParser(description="Generate input YAML manifests")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {_INPUTS_DIR})")
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else None
    generate_all(out)


if __name__ == "__main__":
    main()
