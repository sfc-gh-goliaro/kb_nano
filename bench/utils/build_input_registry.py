#!/usr/bin/env python3
"""Build InputRegistry manifests from workload-derived raw traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from kb_nano import INPUTS_DIR, TRACE_DIR
from kb_nano.bench.utils.trace_schema import DATA_DEPENDENT_OPS, stable_hash


def _read_events(trace_dir: Path) -> list[dict[str, Any]]:
    events = []
    for path in sorted(trace_dir.rglob("*.jsonl")):
        if "flashinfer" in path.parts or path.name.startswith("flashinfer"):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                event["_trace_file"] = str(path)
                events.append(event)
    return events


def _tensor_specs(inputs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, meta in inputs.items():
        if not isinstance(meta, dict):
            continue
        kind = meta.get("kind")
        if kind == "tensor":
            result[name] = {
                "shape": list(meta["shape"]),
                "dtype": meta["dtype"],
            }
        elif kind == "scalar":
            result[name] = meta.get("value")
        elif kind == "sequence" and "value" in meta:
            result[name] = meta["value"]
    return result


def _axis_name(input_name: str, dim_idx: int, dim: int) -> str:
    known = {
        0: "batch_or_tokens",
        1: f"{input_name}_dim1",
        2: f"{input_name}_dim2",
        3: f"{input_name}_dim3",
    }
    if dim in {1, 2, 4, 8, 16, 32, 64, 128}:
        return f"{input_name}_d{dim_idx}"
    return known.get(dim_idx, f"{input_name}_dim{dim_idx}")


def _definition_name(event: dict[str, Any]) -> str:
    init_hash = stable_hash(event.get("init_args", {}))
    return f"{event['op']}_{init_hash}"


def _flashinfer_workload(event: dict[str, Any], scenario_inputs: dict[str, Any]) -> dict[str, Any]:
    axes: dict[str, int] = {}
    workload_inputs: dict[str, Any] = {}
    golden_path = event.get("golden_path")

    for input_name, meta in event.get("inputs", {}).items():
        if not isinstance(meta, dict):
            continue
        if meta.get("kind") == "tensor":
            for dim_idx, dim in enumerate(meta.get("shape", [])):
                axes[_axis_name(input_name, dim_idx, dim)] = int(dim)
            if golden_path:
                workload_inputs[input_name] = {
                    "type": "safetensors",
                    "path": golden_path,
                    "tensor_key": input_name,
                }
            else:
                workload_inputs[input_name] = {"type": "random"}
        elif meta.get("kind") == "scalar":
            workload_inputs[input_name] = {
                "type": "scalar",
                "value": meta.get("value"),
            }
        elif input_name in scenario_inputs and not isinstance(scenario_inputs[input_name], dict):
            workload_inputs[input_name] = {
                "type": "scalar",
                "value": scenario_inputs[input_name],
            }

    workload_uuid = stable_hash({
        "definition": _definition_name(event),
        "inputs": scenario_inputs,
        "golden_path": golden_path,
    })
    return {
        "definition": _definition_name(event),
        "workload": {
            "uuid": workload_uuid,
            "axes": axes,
            "inputs": workload_inputs,
        },
        "solution": None,
        "evaluation": None,
        "kb_nano": {
            "op": event["op"],
            "model_key": event["model_key"],
            "model": event["model"],
            "tp": event["tp"],
            "dtype": event["dtype"],
            "workload": event["workload"],
            "module_path": event["module_path"],
            "observed_count": event.get("_observed_count", 1),
            "source_trace": event.get("_trace_file"),
            "signature": event.get("signature"),
        },
    }


def _primary_size(event: dict[str, Any]) -> int:
    best = 0
    for meta in event.get("inputs", {}).values():
        if isinstance(meta, dict) and meta.get("kind") == "tensor":
            shape = meta.get("shape", [])
            if shape:
                best = max(best, int(shape[0]))
            best = max(best, int(meta.get("numel", 0)))
    return best


def _group_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["op"],
        event.get("model_key"),
        event.get("tp"),
        event.get("dtype"),
        stable_hash(event.get("init_args", {})),
    )


def _select_representatives(events: list[dict[str, Any]], max_per_group: int) -> list[dict[str, Any]]:
    by_signature: dict[str, dict[str, Any]] = {}
    for event in events:
        sig = event.get("signature")
        if sig in by_signature:
            by_signature[sig]["_observed_count"] += 1
            if not by_signature[sig].get("golden_path") and event.get("golden_path"):
                by_signature[sig]["golden_path"] = event["golden_path"]
            continue
        item = dict(event)
        item["_observed_count"] = 1
        by_signature[sig] = item

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in by_signature.values():
        grouped[_group_key(event)].append(event)

    selected = []
    for group_events in grouped.values():
        ordered = sorted(group_events, key=lambda e: (_primary_size(e), e.get("signature", "")))
        if len(ordered) <= max_per_group:
            selected.extend(ordered)
            continue
        picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        remaining = [e for e in ordered if e not in picks]
        remaining.sort(key=lambda e: e.get("_observed_count", 0), reverse=True)
        for event in remaining:
            if len(picks) >= max_per_group:
                break
            picks.append(event)
        selected.extend(sorted(picks, key=lambda e: (_primary_size(e), e.get("signature", ""))))
    return selected


def _scenario_name(event: dict[str, Any]) -> str:
    size = _primary_size(event)
    phase = "decode" if size <= 128 else "prefill"
    suffix = stable_hash({
        "signature": event.get("signature"),
        "module_path": event.get("module_path"),
    })[:8]
    return (
        f"{event['model_key']}/{event['workload']}/{phase}-{size}/"
        f"tp{event['tp']}/{event['dtype']}/{suffix}"
    )


def build_registry(
    *,
    trace_dir: Path,
    output_dir: Path,
    flashinfer_out: Path,
    max_per_group: int = 6,
) -> tuple[Path, Path]:
    events = _read_events(trace_dir)
    if not events:
        raise RuntimeError(f"No trace events found under {trace_dir}")
    selected = _select_representatives(events, max_per_group=max_per_group)

    manifest: dict[str, dict[str, list[dict[str, Any]]]] = {}
    workload_traces = []
    for event in selected:
        op = event["op"]
        inputs = _tensor_specs(event.get("inputs", {}))
        scenario: dict[str, Any] = {
            "name": _scenario_name(event),
            "init_args": event.get("init_args", {}),
            "inputs": inputs,
            "source": {
                "trace": event.get("_trace_file"),
                "signature": event.get("signature"),
                "observed_count": event.get("_observed_count", 1),
                "module_path": event.get("module_path"),
            },
        }
        if op in DATA_DEPENDENT_OPS and event.get("golden_path"):
            scenario["golden"] = event["golden_path"]
        manifest.setdefault(op, {"scenarios": []})["scenarios"].append(scenario)
        workload_traces.append(_flashinfer_workload(event, inputs))

    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "generated.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    flashinfer_out.parent.mkdir(parents=True, exist_ok=True)
    with open(flashinfer_out, "w") as f:
        for row in workload_traces:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    return yaml_path, flashinfer_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build InputRegistry manifests from raw traces")
    parser.add_argument("--trace-dir", type=str, default=str(TRACE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(INPUTS_DIR))
    parser.add_argument(
        "--flashinfer-out",
        type=str,
        default=str(TRACE_DIR / "flashinfer_workloads.jsonl"),
    )
    parser.add_argument("--max-per-group", type=int, default=6)
    args = parser.parse_args()

    yaml_path, flashinfer_path = build_registry(
        trace_dir=Path(args.trace_dir),
        output_dir=Path(args.output_dir),
        flashinfer_out=Path(args.flashinfer_out),
        max_per_group=args.max_per_group,
    )
    print(f"Wrote InputRegistry manifest: {yaml_path}")
    print(f"Wrote FlashInfer-style workloads: {flashinfer_path}")


if __name__ == "__main__":
    main()
