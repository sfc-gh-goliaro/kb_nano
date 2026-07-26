#!/usr/bin/env python3
"""Materialize the AKO-shaped kb dataset (definitions/ + workloads/) for kb_rms_norm.

Source of truth = the kb shape registry
(`bench/kernels/benchmark_scenarios/small/shape_registry.yaml`, operator
`rms_norm`); the uuid of each workload IS the kb scenario name, so
`--scenarios <uuid,...>` on the entrypoint selects exactly the requested
workloads.

Output layout (what spawn.py globs):
    <out>/definitions/kb/kb_rms_norm.json
    <out>/workloads/kb/kb_rms_norm.jsonl

Run with the kb main venv (needs pyyaml):
    python make_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_REPO = Path(os.environ.get("KB_REPO") or HERE.parents[2])
DEFAULT_REGISTRY = (_REPO / "bench" / "kernels" / "benchmark_scenarios"
                    / "small" / "shape_registry.yaml")
DEFAULT_OUT = Path(os.environ.get("KB_TRACE_DIR")
                   or (Path(os.environ["AGENTS_DIR"]) / "kb-trace"
                       if os.environ.get("AGENTS_DIR")
                       else _REPO / "build" / "kb-trace"))
OP = "rms_norm"
DEF_NAME = f"kb_{OP}"
OP_TYPE = "kb"


def build_definition(scenarios: list[dict], reference_src: str) -> dict:
    hidden = sorted({s["init_args"]["hidden_size"] for s in scenarios})
    tokens = sorted({s["inputs"]["x"]["shape"][0] for s in scenarios})
    dtypes = sorted({v["dtype"] for s in scenarios for v in s["inputs"].values()
                     if isinstance(v, dict) and "dtype" in v})
    eps = sorted({s["init_args"]["eps"] for s in scenarios})
    n_res = sum(1 for s in scenarios if "residual" in s["inputs"])

    description = (
        "RMSNorm over the last dimension of a 2-D [tokens, hidden] activation "
        "tensor, as used by the kb-nano/fastkernels L1 baseline "
        f"(tasks/baseline/L1/rms_norm.py). {len(scenarios)} scenarios taken "
        f"verbatim from the kb shape registry (operator '{OP}', small suite): "
        f"hidden sizes {hidden}, token counts {tokens[0]}-{tokens[-1]}, "
        f"eps in {eps}, dtype {'/'.join(dtypes)} for every tensor in every "
        "scenario (the kb registry declares no fp16 or fp32 rms_norm scenario). "
        f"{n_res} of the {len(scenarios)} scenarios also pass a `residual` "
        "tensor: that path is the fused add+norm, and it is IN-PLACE — "
        "`residual := x + residual` and `x := rmsnorm(residual) * weight`, "
        "returning `(x, residual)`. The kb harness compares mutated inputs as "
        "well as returned outputs, so the mutation is part of the contract.\n\n"
        "Interface contract (frozen — the harness copies the baseline's "
        "state_dict into your module and calls it by keyword):\n"
        "  class RMSNorm(nn.Module)\n"
        "    __init__(self, hidden_size, eps=1e-6, elementwise_affine=True, "
        "**kwargs)\n"
        "    forward(self, x, residual=None) -> Tensor | (Tensor, Tensor)\n"
        "  parameter name `weight` (shape [hidden_size]) when "
        "elementwise_affine, else a non-persistent `_unit_weight` buffer.\n"
        "Renaming/adding/removing parameters fails the scenario with "
        "RUNTIME_ERROR (weight_transfer_incomplete) before any numerics run."
    )

    return {
        "name": DEF_NAME,
        "op_type": OP_TYPE,
        "description": description,
        "axes": {
            "tokens": {
                "type": "var",
                "description": "rows of the activation tensor (x.shape[0]); "
                               "the grouping axis for scoring",
                "values": tokens,
            },
            "hidden": {
                "type": "var",
                "description": "normalized dimension (x.shape[1]) == hidden_size",
                "values": hidden,
            },
            "eps": {
                "type": "var",
                "description": "variance epsilon passed to __init__",
                "values": eps,
            },
            "residual": {
                "type": "var",
                "description": "whether the scenario passes a residual tensor "
                               "(fused add+norm, in-place)",
                "values": [False, True],
            },
        },
        "inputs": {
            "x": {"dtype": "bfloat16", "shape": ["tokens", "hidden"],
                  "description": "activations; mutated in place on the residual path"},
            "residual": {"dtype": "bfloat16", "shape": ["tokens", "hidden"],
                         "optional": True,
                         "description": "residual stream; mutated in place "
                                        "(residual := x + residual)"},
        },
        "outputs": {
            "out": {"dtype": "bfloat16", "shape": ["tokens", "hidden"],
                    "description": "normalized activations; on the residual path "
                                   "the return is the tuple (x, residual) and both "
                                   "alias the mutated inputs"},
        },
        "reference": reference_src,
    }


def build_workloads(scenarios: list[dict]) -> list[dict]:
    lines = []
    for s in scenarios:
        x = s["inputs"]["x"]
        lines.append({
            "definition": DEF_NAME,
            "solution": None,
            "workload": {
                "uuid": s["name"],
                "axes": {
                    "tokens": int(x["shape"][0]),
                    "hidden": int(s["init_args"]["hidden_size"]),
                    "eps": float(s["init_args"]["eps"]),
                    "residual": "residual" in s["inputs"],
                    "dtype": str(x["dtype"]),
                },
                "inputs": {},
            },
            "evaluation": None,
        })
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--reference", type=Path, default=HERE / "seed_kernel.py")
    args = ap.parse_args()

    scenarios = yaml.safe_load(args.registry.read_text())[OP]["scenarios"]
    names = [s["name"] for s in scenarios]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate scenario names in the registry — uuids must be unique")

    definition = build_definition(scenarios, args.reference.read_text())
    workloads = build_workloads(scenarios)

    def_path = args.out / "definitions" / OP_TYPE / f"{DEF_NAME}.json"
    wl_path = args.out / "workloads" / OP_TYPE / f"{DEF_NAME}.jsonl"
    def_path.parent.mkdir(parents=True, exist_ok=True)
    wl_path.parent.mkdir(parents=True, exist_ok=True)
    def_path.write_text(json.dumps(definition, indent=2) + "\n")
    wl_path.write_text("".join(json.dumps(w) + "\n" for w in workloads))

    print(f"wrote {def_path} ({len(definition['reference'].splitlines())}-line reference)")
    print(f"wrote {wl_path} ({len(workloads)} workloads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
