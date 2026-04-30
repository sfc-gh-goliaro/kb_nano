#!/usr/bin/env python3
"""Workload-derived kernel input registry pipeline.

This module backs the public registry commands:
``trace-inputs``, ``build-input-registry``, ``validate-input-registry``,
``generate-inputs``, and ``capture-golden``.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import yaml
from safetensors.torch import save_file
from tqdm.auto import tqdm

from kb_nano import GOLDEN_DIR, INPUTS_DIR, KB_ROOT, TRACE_DIR
from kb_nano.bench.kernels.scenario_registry import InputRegistry
from kb_nano.bench.kernels.scenario_schema import (
    DATA_DEPENDENT_INPUTS,
    DATA_DEPENDENT_OPS,
    TraceEvent,
    flatten_named_values,
    stable_hash,
    summarize_value,
)


# --- trace ---
DEFAULT_PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "Write a Python hello world program.",
    "Describe the water cycle.",
]


def _progress(iterable, *, desc: str, total: int | None = None, unit: str = "it"):
    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True)


def _short_model_key(model_name: str) -> str:
    name = model_name.split("/")[-1].lower()
    if "mixtral" in name:
        return "mixtral-8x7b"
    if "llama" in name:
        if "70b" in name:
            return "llama31-70b"
        if "8b" in name:
            return "llama31-8b"
        return "llama31"
    return name.replace("_", "-")


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


def _dtype_name(dtype: str | None) -> str:
    if not dtype:
        return "auto"
    return dtype.replace("torch.", "")


def _torch_dtype(dtype: str) -> torch.dtype | None:
    return {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(_dtype_name(dtype))


def _model_family_matches(model_key: str, family: str) -> bool:
    return (
        model_key == family
        or model_key.startswith(f"{family}-")
        or family.startswith(f"{model_key}-")
    )


def _resolve_targets(model_key: str) -> dict[type, str]:
    """Import benchmark target classes used by the current model family.

    This uses the same model/operator map as ``python -m kb_nano.bench.kernels
    --map``.  The fallback path is only for environments where full discovery
    fails because of optional dependencies.
    """
    try:
        from kb_nano.infra.kernel_swapper import discover_targets

        targets = [
            t for t in discover_targets()
            if any(_model_family_matches(model_key, family) for family in t.models)
        ]
        if targets:
            return {t.target_cls: t.name for t in targets}
        print(
            f"WARNING: no mapped benchmark targets found for model_key={model_key!r}; "
            "falling back to all discovered targets"
        )
        return {t.target_cls: t.name for t in discover_targets()}
    except Exception as exc:
        print(f"WARNING: full target discovery failed: {exc}")

    result: dict[type, str] = {}
    try:
        import importlib
        import torch.nn as nn

        from kb_nano import KB_ROOT

        for level in ("L1", "L2", "L3", "L4"):
            level_dir = KB_ROOT / "tasks" / "baseline" / level
            for path in sorted(level_dir.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                module_name = f"kb_nano.tasks.baseline.{level}.{path.stem}"
                try:
                    mod = importlib.import_module(module_name)
                except Exception:
                    continue
                target_cls = None
                for value in vars(mod).values():
                    if (
                        isinstance(value, type)
                        and issubclass(value, nn.Module)
                        and value is not nn.Module
                        and value.__module__ == mod.__name__
                    ):
                        target_cls = value
                if target_cls is not None:
                    result[target_cls] = path.stem
    except Exception as exc:
        print(f"WARNING: tolerant target discovery failed: {exc}")
    return result


def _bind_forward_inputs(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(module.forward)
        bound = sig.bind_partial(*args, **kwargs)
        return dict(bound.arguments)
    except Exception:
        result = {f"arg{i}": value for i, value in enumerate(args)}
        result.update(kwargs)
        return result


def _summarize_outputs(output: Any) -> Any:
    return summarize_value(output)


def _capture_name_matches(name: str, selected: set[str]) -> bool:
    return name in selected or any(name.startswith(f"{prefix}.") for prefix in selected)


def _module_path_lookup(root: torch.nn.Module) -> dict[int, str]:
    return {id(module): name for name, module in root.named_modules()}


def _extract_init_args(module: torch.nn.Module) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float, bool, str)) for v in value):
            result[name] = list(value)
    return result


class InputTraceRecorder:
    def __init__(
        self,
        *,
        model: str,
        model_key: str,
        tp: int,
        dtype: str,
        workload: str,
        trace_path: Path,
        golden_dir: Path,
        ops: set[str] | None = None,
        capture_golden: bool = True,
    ):
        self.model = model
        self.model_key = model_key
        self.tp = tp
        self.dtype = dtype
        self.workload = workload
        self.trace_path = trace_path
        self.golden_dir = golden_dir
        self.ops = ops
        self.capture_golden = capture_golden
        self.target_classes = _resolve_targets(model_key)
        self.occurrences: dict[str, int] = defaultdict(int)
        self.seen_goldens: set[str] = set()
        self.events_written = 0
        self.goldens_captured = 0
        self.handles = []
        self._file = None

    def __enter__(self) -> "InputTraceRecorder":
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.trace_path, "w")
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._file is not None:
            self._file.close()
            self._file = None

    def attach(self, model: torch.nn.Module) -> int:
        paths = _module_path_lookup(model)
        count = 0
        modules = list(model.modules())
        for module in _progress(
            modules,
            desc=f"Scanning modules for {self.model_key}/{self.workload}",
            unit="module",
        ):
            op = self._op_name(module)
            if op is None:
                continue
            if self.ops is not None and op not in self.ops:
                continue
            module_path = paths.get(id(module), "")
            handle = module.register_forward_hook(
                self._make_hook(op, module_path),
                with_kwargs=True,
            )
            self.handles.append(handle)
            count += 1
        return count

    def _op_name(self, module: torch.nn.Module) -> str | None:
        for cls, op_name in self.target_classes.items():
            if isinstance(module, cls):
                return op_name
        return None

    def _make_hook(self, op: str, module_path: str):
        def hook(module, args, kwargs, output):
            self.occurrences[op] += 1
            occurrence = self.occurrences[op]
            named_inputs = _bind_forward_inputs(module, args, kwargs or {})
            inputs_meta = {
                name: summarize_value(value)
                for name, value in named_inputs.items()
            }
            event = TraceEvent(
                op=op,
                model_key=self.model_key,
                model=self.model,
                tp=self.tp,
                dtype=self.dtype,
                workload=self.workload,
                module_path=module_path,
                module_class=module.__class__.__name__,
                occurrence=occurrence,
                first_occurrence=occurrence == 1,
                inputs=inputs_meta,
                init_args=_extract_init_args(module),
                outputs=_summarize_outputs(output),
            )
            if self.capture_golden and op in DATA_DEPENDENT_OPS:
                event.golden_path = self._capture_golden(op, event, named_inputs)
            assert self._file is not None
            self._file.write(event.to_json() + "\n")
            self.events_written += 1
        return hook

    def _capture_golden(self, op: str, event: TraceEvent, inputs: dict[str, Any]) -> str | None:
        signature = event.canonical_key()
        first_key = f"{op}:{signature}"
        if first_key in self.seen_goldens:
            return None
        self.seen_goldens.add(first_key)

        flat = flatten_named_values(inputs)
        selected_inputs = DATA_DEPENDENT_INPUTS.get(op, set())
        if selected_inputs:
            flat = {
                name: value
                for name, value in flat.items()
                if _capture_name_matches(name, selected_inputs)
            }
        tensors = {
            name: value.detach().cpu().contiguous()
            for name, value in flat.items()
            if isinstance(value, torch.Tensor)
        }
        scalars = {
            name: value
            for name, value in flat.items()
            if not isinstance(value, torch.Tensor)
        }
        if not tensors:
            return None

        scenario_id = stable_hash({
            "op": op,
            "signature": signature,
            "module_path": event.module_path,
        })
        rel = Path(self.model_key) / f"tp{self.tp}" / self.dtype / op / f"{scenario_id}.safetensors"
        out_path = self.golden_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "op": op,
            "signature": signature,
            "model_key": self.model_key,
            "tp": str(self.tp),
            "dtype": self.dtype,
            "module_path": event.module_path,
            "captured_inputs": ",".join(sorted(selected_inputs)),
        }
        save_file(tensors, out_path, metadata=metadata)
        if scalars:
            with open(out_path.with_suffix(".json"), "w") as f:
                json.dump(scalars, f, sort_keys=True, default=str)
        self.goldens_captured += 1
        print(f"Captured input tensors for {op}: {rel}")
        return str(rel)


def trace_llm_model(
    *,
    model_name: str,
    model_key: str | None,
    tp: int,
    dtype: str,
    workload: str,
    prompts: list[str | list[int]] | None,
    output_lens: list[int] | None,
    trace_dir: Path,
    golden_dir: Path,
    ops: set[str] | None = None,
    seed: int = 42,
    enforce_eager: bool = True,
    capture_golden: bool = True,
    num_requests: int | None = None,
    decode_cap: int | None = None,
) -> Path:
    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    key = model_key or _short_model_key(model_name)
    trace_path = trace_dir / key / f"tp{tp}" / dtype / f"{workload}.jsonl"
    print(
        f"Tracing {key}/{workload}: model={model_name}, tp={tp}, dtype={dtype}, "
        f"ops={','.join(sorted(ops)) if ops else 'mapped'}"
    )
    print(f"Raw trace output: {trace_path}")
    engine = LlamaEngine(
        model_name=model_name,
        dtype=_torch_dtype(dtype),
        seed=seed,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp,
    )
    try:
        recorder = InputTraceRecorder(
            model=model_name,
            model_key=key,
            tp=tp,
            dtype=_dtype_name(dtype),
            workload=workload,
            trace_path=trace_path,
            golden_dir=golden_dir,
            ops=ops,
            capture_golden=capture_golden,
        )
        with recorder:
            model = getattr(engine, "model", None)
            if model is None:
                model = engine.model_runner.model
            attached = recorder.attach(model)
            print(f"Attached {attached} trace hooks for {model_name} ({workload})")
            if prompts is None or output_lens is None:
                print(f"Loading standardized workload {workload}...")
                prompts, output_lens = _load_standard_workload(
                    workload,
                    engine.tokenizer,
                    num_requests=num_requests,
                    decode_cap=decode_cap,
                    seed=seed,
                )
            total_prompt_tokens = sum(len(prompt) for prompt in prompts)
            total_output_tokens = sum(output_lens)
            print(
                f"Running generation for {len(prompts)} requests "
                f"({total_prompt_tokens} prompt tokens, {total_output_tokens} max output tokens)..."
            )
            sp_list = [
                SamplingParams(
                    temperature=0.0,
                    max_tokens=out_len,
                    ignore_eos=True,
                )
                for out_len in output_lens
            ]
            engine.generate(prompts, sp_list if len(sp_list) > 1 else sp_list[0])
            print(
                f"Finished {key}/{workload}: wrote {recorder.events_written} events, "
                f"captured {recorder.goldens_captured} input blobs"
            )
    finally:
        if hasattr(engine, "_cleanup"):
            engine._cleanup()
            try:
                atexit.unregister(engine._cleanup)
            except Exception:
                pass
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return trace_path


def _load_standard_workload(
    workload: str,
    tokenizer: Any,
    *,
    num_requests: int | None,
    decode_cap: int | None,
    seed: int,
) -> tuple[list[list[int]], list[int]]:
    from kb_nano.bench.utils.real_prompts import load_real_prompt_workload
    from kb_nano.bench.utils.workloads import THROUGHPUT_WORKLOADS

    by_name = {w.name: w for w in THROUGHPUT_WORKLOADS}
    if workload not in by_name:
        available = ", ".join(sorted(by_name))
        raise ValueError(
            f"Workload {workload!r} does not provide prompts/output_lens and "
            f"is not a standardized throughput workload. Available: {available}"
        )
    spec = by_name[workload]
    n = num_requests if num_requests is not None else min(spec.num_requests, 4)
    samples = load_real_prompt_workload(
        spec.name,
        tokenizer,
        num_requests=n,
        decode_cap=decode_cap,
        dataset_name=spec.dataset_name,
        seed=seed,
    )
    return (
        [sample.prompt_token_ids for sample in samples],
        [sample.output_len for sample in samples],
    )


def _iter_model_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for item in config.get("models", []):
        model = item["hf_name"] if "hf_name" in item else item["model"]
        model_key = item.get("key")
        tp = int(item.get("tp", 1))
        dtype = _dtype_name(item.get("dtype", "bfloat16"))
        prompts = item.get("prompts")
        output_lens = item.get("output_lens")
        workloads = item.get("workloads") or [{
            "name": "trace",
            "prompts": prompts or DEFAULT_PROMPTS,
            "output_lens": output_lens,
        }]
        for workload in workloads:
            if isinstance(workload, str):
                jobs.append({
                    "model": model,
                    "model_key": model_key,
                    "tp": tp,
                    "dtype": dtype,
                    "workload": workload,
                    "prompts": prompts,
                    "output_lens": output_lens,
                    "num_requests": item.get("num_requests"),
                    "decode_cap": item.get("decode_cap"),
                })
            else:
                wl_prompts = workload.get("prompts", prompts)
                wl_output_lens = workload.get("output_lens", output_lens)
                if wl_prompts is not None and wl_output_lens is None:
                    wl_output_lens = [32 for _ in wl_prompts]
                jobs.append({
                    "model": model,
                    "model_key": model_key,
                    "tp": tp,
                    "dtype": dtype,
                    "workload": workload.get("name", "trace"),
                    "prompts": wl_prompts,
                    "output_lens": wl_output_lens,
                    "num_requests": workload.get("num_requests", item.get("num_requests")),
                    "decode_cap": workload.get("decode_cap", item.get("decode_cap")),
                })
    return jobs


def trace_inputs_main() -> None:
    parser = argparse.ArgumentParser(description="Trace operator input metadata from real workloads")
    parser.add_argument("--config", type=str, required=True, help="YAML/JSON workload scenario file")
    parser.add_argument("--trace-dir", type=str, default=str(TRACE_DIR))
    parser.add_argument("--golden-dir", type=str, default=str(GOLDEN_DIR))
    parser.add_argument("--ops", nargs="*", default=None, help="Optional operator-name allowlist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-golden", action="store_true", help="Do not capture first-seen data-dependent tensors")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    jobs = _iter_model_jobs(cfg)
    if not jobs:
        raise SystemExit("No model jobs found in workload scenario file")
    print(f"Loaded {len(jobs)} trace job(s) from {args.config}")
    for idx, job in enumerate(jobs, start=1):
        print(
            f"  [{idx}/{len(jobs)}] {job['model_key'] or _short_model_key(job['model'])}/"
            f"{job['workload']} tp={job['tp']} dtype={job['dtype']} "
            f"num_requests={job.get('num_requests')}"
        )

    trace_dir = Path(args.trace_dir)
    golden_dir = Path(args.golden_dir)
    ops = set(args.ops) if args.ops else None

    if len(jobs) > 1 and os.environ.get("KB_NANO_TRACE_CHILD") != "1":
        for job in _progress(jobs, desc="Trace jobs", unit="job"):
            child_cfg = {
                "models": [{
                    "key": job["model_key"],
                    "hf_name": job["model"],
                    "tp": job["tp"],
                    "dtype": job["dtype"],
                    "workloads": [{
                        "name": job["workload"],
                        "prompts": job["prompts"],
                        "output_lens": job["output_lens"],
                        "num_requests": job.get("num_requests"),
                        "decode_cap": job.get("decode_cap"),
                    }],
                }]
            }
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(child_cfg, f, sort_keys=False)
                child_config = f.name
            cmd = [
                sys.executable, str(Path(__file__).resolve()), "trace-inputs",
                "--config", child_config,
                "--trace-dir", str(trace_dir),
                "--golden-dir", str(golden_dir),
                "--seed", str(args.seed),
            ]
            if args.no_golden:
                cmd.append("--no-golden")
            if args.ops:
                cmd.append("--ops")
                cmd.extend(args.ops)
            env = dict(os.environ)
            env["KB_NANO_TRACE_CHILD"] = "1"
            try:
                print(
                    f"Starting child trace job: "
                    f"{job['model_key'] or _short_model_key(job['model'])}/{job['workload']}"
                )
                subprocess.run(cmd, check=True, env=env)
            finally:
                try:
                    os.unlink(child_config)
                except OSError:
                    pass
        return

    for job in _progress(jobs, desc="Trace jobs", unit="job"):
        out = trace_llm_model(
            model_name=job["model"],
            model_key=job["model_key"],
            tp=job["tp"],
            dtype=job["dtype"],
            workload=job["workload"],
            prompts=job["prompts"],
            output_lens=job["output_lens"],
            trace_dir=trace_dir,
            golden_dir=golden_dir,
            ops=ops,
            seed=args.seed,
            capture_golden=not args.no_golden,
            num_requests=job.get("num_requests"),
            decode_cap=job.get("decode_cap"),
        )
        print(f"Wrote trace: {out}")

# --- build ---

def _read_events(trace_dir: Path) -> list[dict[str, Any]]:
    events = []
    trace_files = [
        path for path in sorted(trace_dir.rglob("*.jsonl"))
        if path.name not in {"flashinfer_workloads.jsonl", "representative_workloads.jsonl"}
    ]
    print(f"Reading raw trace events from {len(trace_files)} file(s) under {trace_dir}")
    for path in _progress(trace_files, desc="Trace files", unit="file"):
        if path.name in {"flashinfer_workloads.jsonl", "representative_workloads.jsonl"}:
            continue
        read_count = 0
        kept_count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                read_count += 1
                event = json.loads(line)
                if event.get("op") in DATA_DEPENDENT_OPS:
                    golden_path = event.get("golden_path")
                    if not golden_path or not (GOLDEN_DIR / golden_path).exists():
                        continue
                event["_trace_file"] = str(path)
                events.append(event)
                kept_count += 1
        print(f"  {path}: kept {kept_count}/{read_count} events")
    print(f"Loaded {len(events)} trace events")
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
    init_hash = stable_hash(_structural_init_args(event.get("init_args", {})))
    return f"{event['op']}_{init_hash}"


def _structural_init_args(init_args: dict[str, Any]) -> dict[str, Any]:
    """Drop scalar constants that do not change the scenario structure.

    Constants such as RMSNorm ``eps`` should not create separate registry
    scenarios when all input shapes and structural fields are identical.
    """
    result: dict[str, Any] = {}
    for key, value in sorted(init_args.items()):
        if isinstance(value, float):
            continue
        if isinstance(value, dict):
            nested = _structural_init_args(value)
            if nested:
                result[key] = nested
        elif isinstance(value, (list, tuple)):
            kept = [v for v in value if not isinstance(v, float)]
            if kept:
                result[key] = kept
        else:
            result[key] = value
    return result


def _canonical_for_scenario(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            return {
                "kind": "tensor",
                "shape": value.get("shape"),
                "dtype": value.get("dtype"),
                "stride": value.get("stride"),
                "layout": value.get("layout"),
            }
        if value.get("kind") in {"scalar", "sequence"}:
            return value
        return {str(k): _canonical_for_scenario(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_for_scenario(v) for v in value]
    return value


def _scenario_signature(event: dict[str, Any]) -> str:
    return stable_hash({
        "op": event["op"],
        "inputs": _canonical_for_scenario(event.get("inputs", {})),
        "init_args": _canonical_for_scenario(
            _structural_init_args(event.get("init_args", {}))
        ),
    })


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
    # Use the leading activation/routing axis for scenario naming and
    # representative selection. Parameter tensors can dominate numel by orders
    # of magnitude and produce useless names like prefill-469762048.
    preferred = ("x", "A", "topk_ids", "hidden_states", "input")
    for name in preferred:
        meta = event.get("inputs", {}).get(name)
        if isinstance(meta, dict) and meta.get("kind") == "tensor":
            shape = meta.get("shape", [])
            if shape:
                return int(shape[0])

    best = 0
    for meta in event.get("inputs", {}).values():
        if isinstance(meta, dict) and meta.get("kind") == "tensor":
            shape = meta.get("shape", [])
            if shape:
                best = max(best, int(shape[0]))
    return best


def _shape_family_key(event: dict[str, Any]) -> str:
    primary = _primary_size(event)
    token_dependent = {"expert_ids", "sorted_token_ids"}
    family_inputs: dict[str, Any] = {}

    for name, meta in sorted(event.get("inputs", {}).items()):
        if not isinstance(meta, dict):
            continue
        kind = meta.get("kind")
        if kind == "tensor":
            shape = list(meta.get("shape", []))
            if shape:
                if name in token_dependent:
                    shape[0] = "token_dependent"
                elif not meta.get("requires_grad"):
                    dim0 = int(shape[0])
                    if primary > 0 and dim0 % primary == 0:
                        shape[0] = ["tokens", dim0 // primary]
                    else:
                        shape[0] = "tokens"
            family_inputs[name] = {
                "kind": "tensor",
                "shape": shape,
                "dtype": meta.get("dtype"),
                "layout": meta.get("layout"),
            }
        elif kind in {"scalar", "sequence"}:
            family_inputs[name] = meta

    return stable_hash({
        "op": event["op"],
        "init_args": _structural_init_args(event.get("init_args", {})),
        "inputs": family_inputs,
    })


def _dtype_key(event: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, meta.get("dtype"))
        for name, meta in sorted(event.get("inputs", {}).items())
        if isinstance(meta, dict) and meta.get("kind") == "tensor"
    )


def _group_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["op"],
        _dtype_key(event),
        stable_hash(_structural_init_args(event.get("init_args", {}))),
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _best_for_target(
    events: list[dict[str, Any]],
    target: int,
) -> dict[str, Any]:
    return min(
        events,
        key=lambda e: (
            not _is_power_of_two(_primary_size(e)),
            abs(_primary_size(e) - target),
            -e.get("_observed_count", 0),
            e.get("_scenario_signature", ""),
        ),
    )


def _pick_from_family(events: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda e: (_primary_size(e), e.get("_scenario_signature", "")))
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [max(
            ordered,
            key=lambda e: (
                _is_power_of_two(_primary_size(e)),
                e.get("_observed_count", 0),
                -_primary_size(e),
            ),
        )]

    token_sizes = [_primary_size(event) for event in ordered]
    anchor_indices = [0, len(ordered) - 1] if count == 2 else [
        0,
        len(ordered) // 2,
        len(ordered) - 1,
    ]
    picks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx in anchor_indices:
        available = [
            event for event in ordered
            if event["_scenario_signature"] not in seen
        ]
        if not available or len(picks) >= count:
            break
        pick = _best_for_target(available, token_sizes[idx])
        picks.append(pick)
        seen.add(pick["_scenario_signature"])

    remaining = [
        event for event in ordered
        if event["_scenario_signature"] not in seen
    ]
    remaining.sort(
        key=lambda e: (
            not _is_power_of_two(_primary_size(e)),
            -e.get("_observed_count", 0),
            _primary_size(e),
            e.get("_scenario_signature", ""),
        )
    )
    for event in remaining:
        if len(picks) >= count:
            break
        picks.append(event)
    return sorted(picks, key=lambda e: (_primary_size(e), e.get("_scenario_signature", "")))


def _allocate_family_slots(
    families: dict[str, list[dict[str, Any]]],
    max_shape_families: int,
    max_tokens_per_family: int,
) -> dict[str, int]:
    ranked = sorted(
        families,
        key=lambda key: (
            -sum(e.get("_observed_count", 0) for e in families[key]),
            key,
        ),
    )
    chosen = ranked[:max_shape_families]
    return {
        key: min(max_tokens_per_family, len(families[key]))
        for key in chosen
    }


def _select_representatives(
    events: list[dict[str, Any]],
    *,
    max_shape_families: int,
    max_tokens_per_family: int,
) -> list[dict[str, Any]]:
    by_signature: dict[str, dict[str, Any]] = {}
    print(f"Deduplicating {len(events)} trace events by scenario signature...")
    for event in _progress(events, desc="Deduplicate events", unit="event"):
        sig = _scenario_signature(event)
        if sig in by_signature:
            by_signature[sig]["_observed_count"] += 1
            if not by_signature[sig].get("golden_path") and event.get("golden_path"):
                by_signature[sig]["golden_path"] = event["golden_path"]
            continue
        item = dict(event)
        item["_scenario_signature"] = sig
        item["_observed_count"] = 1
        by_signature[sig] = item
    print(f"Found {len(by_signature)} unique scenario signatures")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in _progress(
        by_signature.values(),
        desc="Group scenarios",
        total=len(by_signature),
        unit="scenario",
    ):
        grouped[_group_key(event)].append(event)
    print(f"Condensing {len(grouped)} op/dtype/init group(s)")

    selected = []
    for group_events in _progress(grouped.values(), desc="Select representatives", unit="group"):
        ordered = sorted(group_events, key=lambda e: (_primary_size(e), e.get("_scenario_signature", "")))
        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in ordered:
            families[_shape_family_key(event)].append(event)
        slots = _allocate_family_slots(
            families,
            max_shape_families=max_shape_families,
            max_tokens_per_family=max_tokens_per_family,
        )
        picks = []
        for family_key, count in slots.items():
            picks.extend(_pick_from_family(families[family_key], count))
        selected.extend(sorted(picks, key=lambda e: (_primary_size(e), e.get("_scenario_signature", ""))))
    print(f"Selected {len(selected)} representative scenario(s)")
    return selected


def _scenario_name(event: dict[str, Any]) -> str:
    size = _primary_size(event)
    suffix = event.get("_scenario_signature") or _scenario_signature(event)
    return f"tokens-{size}/{suffix[:8]}"


def build_registry(
    *,
    trace_dir: Path,
    output_dir: Path,
    flashinfer_out: Path,
    max_shape_families: int = 5,
    max_tokens_per_family: int = 5,
) -> tuple[Path, Path]:
    events = _read_events(trace_dir)
    if not events:
        raise RuntimeError(f"No trace events found under {trace_dir}")
    selected = _select_representatives(
        events,
        max_shape_families=max_shape_families,
        max_tokens_per_family=max_tokens_per_family,
    )

    manifest: dict[str, dict[str, list[dict[str, Any]]]] = {}
    workload_traces = []
    for event in _progress(selected, desc="Write scenario records", unit="scenario"):
        op = event["op"]
        inputs = _tensor_specs(event.get("inputs", {}))
        scenario: dict[str, Any] = {
            "name": _scenario_name(event),
            "init_args": event.get("init_args", {}),
            "inputs": inputs,
        }
        if op in DATA_DEPENDENT_OPS and event.get("golden_path"):
            scenario["golden"] = event["golden_path"]
            scenario["golden_inputs"] = sorted(DATA_DEPENDENT_INPUTS.get(op, set()))
        manifest.setdefault(op, {"scenarios": []})["scenarios"].append(scenario)
        workload_traces.append(_flashinfer_workload(event, inputs))

    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "shape_registry.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    flashinfer_out.parent.mkdir(parents=True, exist_ok=True)
    with open(flashinfer_out, "w") as f:
        for row in workload_traces:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    return yaml_path, flashinfer_out


def build_input_registry_main() -> None:
    parser = argparse.ArgumentParser(description="Build InputRegistry manifests from raw traces")
    parser.add_argument("--trace-dir", type=str, default=str(TRACE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(INPUTS_DIR))
    parser.add_argument(
        "--flashinfer-out",
        type=str,
        default=str(TRACE_DIR / "representative_workloads.jsonl"),
    )
    parser.add_argument("--max-shape-families", type=int, default=5)
    parser.add_argument("--max-tokens-per-family", type=int, default=5)
    args = parser.parse_args()

    yaml_path, flashinfer_path = build_registry(
        trace_dir=Path(args.trace_dir),
        output_dir=Path(args.output_dir),
        flashinfer_out=Path(args.flashinfer_out),
        max_shape_families=args.max_shape_families,
        max_tokens_per_family=args.max_tokens_per_family,
    )
    print(f"Wrote InputRegistry manifest: {yaml_path}")
    print(f"Wrote representative workloads: {flashinfer_path}")

# --- generate ---
_INPUTS_DIR = INPUTS_DIR

METADATA_CANONICAL_M_VALUES = {
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
    for m in METADATA_CANONICAL_M_VALUES["decode"]:
        scenarios.append({
            "name": f"{model_key}/decode-bs{m}/tp{tp}",
            "init_args": {"hidden_size": h, "eps": eps},
            "inputs": {
                "x": {"shape": [m, h], "dtype": "bfloat16"},
                "residual": {"shape": [m, h], "dtype": "bfloat16"},
            },
        })
    for m in METADATA_CANONICAL_M_VALUES["prefill"]:
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
    for m in METADATA_CANONICAL_M_VALUES["decode"]:
        scenarios.append({
            "name": f"{model_key}/decode-bs{m}/tp{tp}",
            "init_args": {},
            "inputs": {"x": {"shape": [m, gate_up], "dtype": "bfloat16"}},
        })
    for m in METADATA_CANONICAL_M_VALUES["prefill"]:
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
    for m in METADATA_CANONICAL_M_VALUES["decode"]:
        scenarios.append({
            "name": f"{model_key}/decode-bs{m}/tp{tp}",
            "init_args": dict(init),
            "inputs": {
                "positions": {"shape": [m], "dtype": "int64"},
                "query": {"shape": [m, nq, head_dim], "dtype": "bfloat16"},
                "key": {"shape": [m, nkv, head_dim], "dtype": "bfloat16"},
            },
        })
    for m in METADATA_CANONICAL_M_VALUES["prefill"]:
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
    for m in METADATA_CANONICAL_M_VALUES["decode"] + METADATA_CANONICAL_M_VALUES["prefill"]:
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


def generate_inputs_main():
    parser = argparse.ArgumentParser(description="Generate input YAML manifests")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {_INPUTS_DIR})")
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else None
    generate_all(out)

# --- capture golden ---
_GOLDEN_DIR = GOLDEN_DIR

GOLDEN_CAPTURE_OPS = [
    "moe_align",
    "moe_grouped_gemm",
    "fused_experts",
    "store_kvcache",
]

GOLDEN_CANONICAL_M_VALUES = [1, 8, 32, 128, 128, 512]

GOLDEN_MODEL_KEY_MAP = {
    "llama": "llama31",
    "mixtral": "mixtral",
}


def _get_model_key(model_name: str) -> str:
    name_lower = model_name.lower()
    for pattern, key in GOLDEN_MODEL_KEY_MAP.items():
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
    print(f"  Data-dependent ops: {GOLDEN_CAPTURE_OPS}")

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


def capture_golden_main():
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

# --- validate ---

def _static_target_names() -> set[str]:
    result: set[str] = set()
    for level in ("L1", "L2", "L3", "L4"):
        level_dir = KB_ROOT / "tasks" / "baseline" / level
        if not level_dir.is_dir():
            continue
        for path in level_dir.glob("*.py"):
            if not path.name.startswith("_"):
                result.add(path.stem)
    return result


def validate_registry(
    *,
    inputs_dir: Path,
    golden_required: bool = True,
    require_all_static_targets: bool = False,
    materialize_inputs: bool = False,
) -> int:
    reg = InputRegistry(inputs_dir=inputs_dir)
    ops = set(reg.operators())
    errors: list[str] = []
    warnings: list[str] = []
    total_scenarios = sum(len(reg.scenarios(op)) for op in ops)
    print(f"Validating {len(ops)} operator(s), {total_scenarios} scenario(s)")

    if require_all_static_targets:
        missing = sorted(_static_target_names() - ops)
        if missing:
            warnings.append(
                f"{len(missing)} static target names have no registry entries; "
                f"sample: {', '.join(missing[:20])}"
            )

    for op in _progress(sorted(ops), desc="Validate operators", unit="op"):
        scenarios = reg.scenarios(op)
        if not scenarios:
            errors.append(f"{op}: no scenarios")
            continue
        for scenario in _progress(
            scenarios,
            desc=f"Validate {op}",
            unit="scenario",
        ):
            if op in DATA_DEPENDENT_OPS and golden_required and not scenario.golden_path:
                errors.append(f"{op}/{scenario.name}: data-dependent scenario has no golden path")
            if scenario.golden_path and golden_required:
                golden_file = GOLDEN_DIR / scenario.golden_path
                if not golden_file.is_file():
                    errors.append(f"{op}/{scenario.name}: golden data file not found: {golden_file}")
            try:
                init_args = reg.get_init_args(op, scenario.name)
            except Exception as exc:
                errors.append(f"{op}/{scenario.name}: cannot load init_args: {exc}")
                continue
            if not isinstance(init_args, dict):
                errors.append(f"{op}/{scenario.name}: init_args is not a mapping")
            if not materialize_inputs:
                scenario_inputs = getattr(scenario, "inputs", None)
                if not isinstance(scenario_inputs, dict):
                    errors.append(f"{op}/{scenario.name}: inputs is not a mapping")
                continue
            try:
                inputs = reg.get_inputs(op, scenario.name, device="cpu")
            except FileNotFoundError as exc:
                if golden_required:
                    errors.append(f"{op}/{scenario.name}: {exc}")
                else:
                    warnings.append(f"{op}/{scenario.name}: missing golden skipped")
                continue
            except Exception as exc:
                errors.append(f"{op}/{scenario.name}: cannot materialize inputs: {exc}")
                continue
            if not isinstance(inputs, dict):
                errors.append(f"{op}/{scenario.name}: inputs did not materialize to a dict")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Validated {len(ops)} operators from {inputs_dir}")
    return 1 if errors else 0


def validate_input_registry_main() -> None:
    parser = argparse.ArgumentParser(description="Validate InputRegistry manifests")
    parser.add_argument("--inputs-dir", type=str, default=str(INPUTS_DIR))
    parser.add_argument("--allow-missing-golden", action="store_true")
    parser.add_argument("--require-all-static-targets", action="store_true")
    parser.add_argument(
        "--materialize-inputs",
        action="store_true",
        help="Instantiate tensors on CPU instead of only validating metadata and captured-input paths",
    )
    args = parser.parse_args()
    raise SystemExit(validate_registry(
        inputs_dir=Path(args.inputs_dir),
        golden_required=not args.allow_missing_golden,
        require_all_static_targets=args.require_all_static_targets,
        materialize_inputs=args.materialize_inputs,
    ))


def main() -> None:
    commands = {
        "generate-inputs": generate_inputs_main,
        "capture-golden": capture_golden_main,
        "trace-inputs": trace_inputs_main,
        "build-input-registry": build_input_registry_main,
        "validate-input-registry": validate_input_registry_main,
    }
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print("Usage: python bench/kernels/scenario_pipeline.py <command> [args...]")
        print()
        print("Commands:")
        for command in commands:
            print(f"  {command}")
        raise SystemExit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    handler = commands.get(command)
    if handler is None:
        print(f"Unknown command: {command}")
        print("Run with --help to list commands.")
        raise SystemExit(1)

    sys.argv = [f"scenario_pipeline.py {command}"] + sys.argv[2:]
    handler()


if __name__ == "__main__":
    main()
