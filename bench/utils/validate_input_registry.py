#!/usr/bin/env python3
"""Validate generated InputRegistry coverage and loadability."""

from __future__ import annotations

import argparse
from pathlib import Path

from kb_nano import INPUTS_DIR
from kb_nano.bench.utils.input_registry import InputRegistry
from kb_nano.bench.utils.trace_schema import DATA_DEPENDENT_OPS


def _static_target_names() -> set[str]:
    root = Path(__file__).resolve().parents[2]
    result: set[str] = set()
    for level in ("L1", "L2", "L3", "L4"):
        level_dir = root / "tasks" / "baseline" / level
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
) -> int:
    reg = InputRegistry(inputs_dir=inputs_dir)
    ops = set(reg.operators())
    errors: list[str] = []
    warnings: list[str] = []

    if require_all_static_targets:
        missing = sorted(_static_target_names() - ops)
        if missing:
            warnings.append(
                f"{len(missing)} static target names have no registry entries; "
                f"sample: {', '.join(missing[:20])}"
            )

    for op in sorted(ops):
        scenarios = reg.scenarios(op)
        if not scenarios:
            errors.append(f"{op}: no scenarios")
            continue
        for scenario in scenarios:
            if op in DATA_DEPENDENT_OPS and golden_required and not scenario.golden_path:
                errors.append(f"{op}/{scenario.name}: data-dependent scenario has no golden path")
            try:
                init_args = reg.get_init_args(op, scenario.name)
            except Exception as exc:
                errors.append(f"{op}/{scenario.name}: cannot load init_args: {exc}")
                continue
            if not isinstance(init_args, dict):
                errors.append(f"{op}/{scenario.name}: init_args is not a mapping")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate InputRegistry manifests")
    parser.add_argument("--inputs-dir", type=str, default=str(INPUTS_DIR))
    parser.add_argument("--allow-missing-golden", action="store_true")
    parser.add_argument("--require-all-static-targets", action="store_true")
    args = parser.parse_args()
    raise SystemExit(validate_registry(
        inputs_dir=Path(args.inputs_dir),
        golden_required=not args.allow_missing_golden,
        require_all_static_targets=args.require_all_static_targets,
    ))


if __name__ == "__main__":
    main()
