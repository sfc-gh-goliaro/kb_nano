"""Unified Input Registry for kernel-level benchmarking.

Loads YAML manifests that describe operator input specifications (shapes, dtypes,
init_args) and optionally references golden .pt files for data-dependent operators.
Provides tensors to the KernelRunner for isolated forward() testing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml

from kb_nano import GOLDEN_DIR, INPUTS_DIR

_INPUTS_DIR = INPUTS_DIR
_GOLDEN_DIR = GOLDEN_DIR

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _parse_dtype(s: str) -> torch.dtype:
    dt = _DTYPE_MAP.get(s)
    if dt is None:
        raise ValueError(f"Unknown dtype {s!r}. Supported: {list(_DTYPE_MAP)}")
    return dt


class Scenario:
    """A single test scenario for an operator."""

    __slots__ = ("name", "init_args", "inputs", "golden_path")

    def __init__(
        self,
        name: str,
        init_args: dict[str, Any],
        inputs: dict[str, Any],
        golden_path: str | None = None,
    ):
        self.name = name
        self.init_args = init_args
        self.inputs = inputs
        self.golden_path = golden_path

    def __repr__(self) -> str:
        return f"Scenario({self.name!r})"


class InputRegistry:
    """Loads and serves operator input specifications from YAML manifests."""

    def __init__(self, inputs_dir: str | Path | None = None, golden_dir: str | Path | None = None):
        self._inputs_dir = Path(inputs_dir) if inputs_dir else _INPUTS_DIR
        self._golden_dir = Path(golden_dir) if golden_dir else _GOLDEN_DIR
        self._operators: dict[str, list[Scenario]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._inputs_dir.is_dir():
            return
        for yaml_file in sorted(self._inputs_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            for op_name, op_spec in data.items():
                scenarios = []
                for s in op_spec.get("scenarios", []):
                    scenarios.append(Scenario(
                        name=s["name"],
                        init_args=s.get("init_args", {}),
                        inputs=s.get("inputs", {}),
                        golden_path=s.get("golden"),
                    ))
                self._operators.setdefault(op_name, []).extend(scenarios)

    def operators(self) -> list[str]:
        """Return all operator names in the registry."""
        self._load()
        return sorted(self._operators.keys())

    def scenarios(
        self,
        operator: str,
        *,
        models: list[str] | None = None,
        tp: list[int] | None = None,
        category: str | None = None,
    ) -> list[Scenario]:
        """Return scenarios for an operator, optionally filtered.

        Filtering is based on scenario name conventions:
          - model filter: scenario name starts with model key (e.g. "llama31-8b/...")
          - tp filter: scenario name contains "/tpN" (e.g. ".../tp4")
        """
        self._load()
        all_scenarios = self._operators.get(operator, [])
        result = all_scenarios

        if models:
            result = [
                s for s in result
                if any(s.name.startswith(m) for m in models)
            ]

        if tp:
            tp_tags = {f"/tp{t}" for t in tp}
            result = [
                s for s in result
                if any(tag in s.name for tag in tp_tags)
            ]

        return result

    def get_inputs(
        self,
        operator: str,
        scenario_name: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Build input tensors for a specific scenario.

        For shape-only inputs, generates random tensors.
        For golden inputs, loads the .pt file.
        Scalar values (int, float) are passed through as-is.
        """
        self._load()
        scenario = None
        for s in self._operators.get(operator, []):
            if s.name == scenario_name:
                scenario = s
                break
        if scenario is None:
            raise KeyError(
                f"Scenario {scenario_name!r} not found for operator {operator!r}"
            )

        if scenario.golden_path:
            golden_file = self._golden_dir / scenario.golden_path
            if not golden_file.is_file():
                raise FileNotFoundError(
                    f"Golden data file not found: {golden_file}\n"
                    f"Run bench/utils/capture_golden.py to generate it, "
                    f"or download from HuggingFace Hub."
                )
            golden_data = torch.load(golden_file, map_location=device, weights_only=True)
            return golden_data

        result: dict[str, Any] = {}
        for arg_name, spec in scenario.inputs.items():
            if isinstance(spec, dict) and "shape" in spec:
                dtype = _parse_dtype(spec["dtype"])
                shape = spec["shape"]
                if dtype in (torch.int32, torch.int64):
                    result[arg_name] = torch.randint(0, 100, shape, dtype=dtype, device=device)
                elif dtype == torch.bool:
                    result[arg_name] = torch.randint(0, 2, shape, dtype=torch.uint8, device=device).bool()
                else:
                    result[arg_name] = torch.randn(shape, dtype=dtype, device=device)
            elif spec is None:
                result[arg_name] = None
            else:
                result[arg_name] = spec
        return result

    def get_init_args(self, operator: str, scenario_name: str) -> dict[str, Any]:
        """Return init_args for a specific scenario."""
        self._load()
        for s in self._operators.get(operator, []):
            if s.name == scenario_name:
                return dict(s.init_args)
        raise KeyError(
            f"Scenario {scenario_name!r} not found for operator {operator!r}"
        )
