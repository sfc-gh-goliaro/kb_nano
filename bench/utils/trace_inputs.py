#!/usr/bin/env python3
"""Trace real model workloads and record operator input metadata.

Raw trace events are kb_nano-specific JSONL records.  They intentionally keep
provenance that is needed for coverage auditing and scenario condensation.
Use ``build-input-registry`` to turn these traces into FlashInfer-style workload
traces plus the legacy YAML consumed by ``InputRegistry``.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

from kb_nano import GOLDEN_DIR, TRACE_DIR
from kb_nano.bench.utils.trace_schema import (
    DATA_DEPENDENT_OPS,
    TraceEvent,
    flatten_named_values,
    stable_hash,
    summarize_value,
)


DEFAULT_PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "Write a Python hello world program.",
    "Describe the water cycle.",
]


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


def _resolve_targets() -> dict[type, str]:
    """Import discoverable benchmark target classes.

    Some target modules have optional dependencies.  If full discovery fails in
    the local environment, tracing can still proceed by matching module class
    names through ``--ops`` filters, but class-based matching is preferred.
    """
    try:
        from kb_nano.infra.kernel_swapper import discover_targets

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
        self.target_classes = _resolve_targets()
        self.occurrences: dict[str, int] = defaultdict(int)
        self.seen_goldens: set[str] = set()
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
        for module in model.modules():
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
        return hook

    def _capture_golden(self, op: str, event: TraceEvent, inputs: dict[str, Any]) -> str | None:
        signature = event.canonical_key()
        first_key = f"{op}:{signature}"
        if first_key in self.seen_goldens:
            return None
        self.seen_goldens.add(first_key)

        flat = flatten_named_values(inputs)
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
        }
        save_file(tensors, out_path, metadata=metadata)
        if scalars:
            with open(out_path.with_suffix(".json"), "w") as f:
                json.dump(scalars, f, sort_keys=True, default=str)
        return str(rel)


def trace_llm_model(
    *,
    model_name: str,
    model_key: str | None,
    tp: int,
    dtype: str,
    workload: str,
    prompts: list[str],
    output_lens: list[int],
    trace_dir: Path,
    golden_dir: Path,
    ops: set[str] | None = None,
    seed: int = 42,
    enforce_eager: bool = True,
    capture_golden: bool = True,
) -> Path:
    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    key = model_key or _short_model_key(model_name)
    trace_path = trace_dir / key / f"tp{tp}" / dtype / f"{workload}.jsonl"
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
            attached = recorder.attach(engine.model)
            print(f"Attached {attached} trace hooks for {model_name} ({workload})")
            sp_list = [
                SamplingParams(
                    temperature=0.0,
                    max_tokens=out_len,
                    ignore_eos=True,
                )
                for out_len in output_lens
            ]
            engine.generate(prompts, sp_list if len(sp_list) > 1 else sp_list[0])
    finally:
        if hasattr(engine, "_cleanup"):
            engine._cleanup()
        del engine
    return trace_path


def _iter_model_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for item in config.get("models", []):
        model = item["hf_name"] if "hf_name" in item else item["model"]
        model_key = item.get("key")
        tp = int(item.get("tp", 1))
        dtype = _dtype_name(item.get("dtype", "bfloat16"))
        prompts = item.get("prompts") or DEFAULT_PROMPTS
        output_lens = item.get("output_lens") or [32 for _ in prompts]
        workloads = item.get("workloads") or [{"name": "trace", "prompts": prompts, "output_lens": output_lens}]
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
                })
            else:
                jobs.append({
                    "model": model,
                    "model_key": model_key,
                    "tp": tp,
                    "dtype": dtype,
                    "workload": workload.get("name", "trace"),
                    "prompts": workload.get("prompts", prompts),
                    "output_lens": workload.get("output_lens", output_lens),
                })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace operator input metadata from real workloads")
    parser.add_argument("--config", type=str, required=True, help="YAML/JSON model trace config")
    parser.add_argument("--trace-dir", type=str, default=str(TRACE_DIR))
    parser.add_argument("--golden-dir", type=str, default=str(GOLDEN_DIR))
    parser.add_argument("--ops", nargs="*", default=None, help="Optional operator-name allowlist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-golden", action="store_true", help="Do not capture first-seen data-dependent tensors")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    jobs = _iter_model_jobs(cfg)
    if not jobs:
        raise SystemExit("No model jobs found in trace config")

    trace_dir = Path(args.trace_dir)
    golden_dir = Path(args.golden_dir)
    ops = set(args.ops) if args.ops else None
    for job in jobs:
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
        )
        print(f"Wrote trace: {out}")


if __name__ == "__main__":
    main()
