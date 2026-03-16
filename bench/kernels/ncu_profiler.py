"""NCU (Nsight Compute) profiling for isolated kernel benchmarks.

Wraps three tasks:
1. Run NCU on a kernel (baseline or candidate) for a given scenario -> CSV.
2. Parse the CSV into a cleaned pandas DataFrame.
3. Convert the DataFrame into a JSON string suitable for LLM/agent prompts.

Adapted from CudaForge/run_ncu.py for kb_nano's nn.Module + InputRegistry
architecture.

Typical usage::

    from kb_nano.bench.kernels.ncu_profiler import profile_scenario

    result = profile_scenario(
        target_name="rms_norm",
        scenario=scenario,
        target=target,
        candidate_path="tasks/candidate/L1/rms_norm.py",
    )
    print(result["baseline_metrics"])   # JSON string
    print(result["candidate_metrics"])  # JSON string
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from kb_nano.bench.utils.input_registry import Scenario
    from kb_nano.infra.kernel_swapper import BenchTarget

__all__ = [
    "METRICS",
    "METRIC_COLUMNS",
    "find_ncu_binary",
    "profile_kernel",
    "load_ncu_metrics",
    "metrics_to_prompt",
    "profile_scenario",
]

# ---------------------------------------------------------------------------
# NCU metric set (23 metrics covering SM, occupancy, compute, memory, cache,
# and warp-stall categories)
# ---------------------------------------------------------------------------
METRICS = ",".join([
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
])

METRIC_COLUMNS: list[str] = [s.strip() for s in METRICS.split(",")]

# ---------------------------------------------------------------------------
# Worker script that NCU profiles.  Instantiates a single nn.Module with
# inputs from the InputRegistry and runs forward() in a loop.
# ---------------------------------------------------------------------------
_NCU_WORKER = r'''
import json, sys, os, torch

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    # Import InputRegistry
    ir_mod = __import__(
        f"{pkg}.bench.utils.input_registry", fromlist=["InputRegistry"]
    )
    registry = ir_mod.InputRegistry()

    # Load the target module class
    if cfg["module_source"] == "baseline":
        mod = __import__(
            f"{pkg}.{cfg['module_path']}", fromlist=[cfg["class_name"]]
        )
        cls = getattr(mod, cfg["class_name"])
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_ncu_target", cfg["module_source"]
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, cfg["class_name"])

    # Instantiate
    init_args = cfg.get("init_args", {})
    try:
        module = cls(**init_args)
    except TypeError:
        module = cls()
    module = module.to("cuda").eval()

    # Load inputs for the scenario
    inputs = registry.get_inputs(
        cfg["target_name"], cfg["scenario_name"], device="cuda"
    )
    tensor_inputs = {
        k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)
    }
    scalar_inputs = {
        k: v for k, v in inputs.items() if not isinstance(v, torch.Tensor)
    }

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            module(**tensor_inputs, **scalar_inputs)
        torch.cuda.synchronize()

    # Profiled iterations (NCU captures these launches)
    num_runs = cfg.get("num_runs", 20)
    with torch.no_grad():
        for _ in range(num_runs):
            module(**tensor_inputs, **scalar_inputs)
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# NCU binary detection
# ---------------------------------------------------------------------------
def find_ncu_binary() -> str | None:
    """Locate the ``ncu`` binary. Returns the path or None if not found."""
    path = shutil.which("ncu")
    if path:
        return path
    for fallback in ("/usr/local/bin/ncu", "/usr/bin/ncu"):
        if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
            return fallback
    return None


# ---------------------------------------------------------------------------
# Core profiling: generate worker, invoke NCU, return CSV path
# ---------------------------------------------------------------------------
def profile_kernel(
    target_name: str,
    scenario_name: str,
    class_name: str,
    module_source: str,
    module_path: str,
    init_args: dict[str, Any],
    num_launches: int = 20,
    out_dir: str | Path | None = None,
    label: str = "",
) -> Path | None:
    """Profile a single kernel with NCU and return the CSV path.

    Parameters
    ----------
    target_name:
        Operator name (e.g. ``"rms_norm"``).
    scenario_name:
        Scenario key from the InputRegistry.
    class_name:
        The ``nn.Module`` class name to instantiate.
    module_source:
        ``"baseline"`` to import via *module_path*, or a file path to a
        candidate ``.py`` file.
    module_path:
        Dotted Python module path for baseline imports
        (e.g. ``"tasks.baseline.L1.rms_norm"``).
    init_args:
        Constructor arguments passed to the module.
    num_launches:
        How many forward() calls NCU will capture.
    out_dir:
        Directory for the output CSV.  Defaults to a temp directory.
    label:
        Human-readable label printed in the ``[ncu]`` log line.

    Returns
    -------
    Path to the CSV file, or ``None`` on failure.
    """
    ncu_bin = find_ncu_binary()
    if ncu_bin is None:
        print("  WARNING: ncu binary not found — skipping NCU profiling. "
              "Install NVIDIA Nsight Compute to enable.")
        return None

    from kb_nano import PROJECT_ROOT

    # Prepare temp working directory
    work_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="kb_ncu_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    worker_file = work_dir / "_ncu_worker.py"
    worker_file.write_text(_NCU_WORKER)

    config = {
        "project_root": str(PROJECT_ROOT),
        "package_name": "kb_nano",
        "target_name": target_name,
        "scenario_name": scenario_name,
        "class_name": class_name,
        "module_source": module_source,
        "module_path": module_path,
        "init_args": init_args,
        "num_runs": num_launches,
    }

    config_file = work_dir / "_ncu_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    safe_name = re.sub(r"[^\w]", "_", f"{target_name}_{label}")
    csv_path = work_dir / f"ncu_{safe_name}.csv"

    # Build NCU command
    cmd = [
        ncu_bin,
        "--csv",
        "--page=raw",
        "--kernel-name-base=demangled",
        "--target-processes=all",
        "--replay-mode=kernel",
        "--profile-from-start=on",
        f"--log-file={csv_path}",
        f"--metrics={METRICS}",
        "--launch-skip=0",
        f"--launch-count={num_launches}",
        sys.executable, str(worker_file), str(config_file),
    ]

    env = os.environ.copy()
    tmp_ncu_dir = Path.home() / "ncu-tmp"
    tmp_ncu_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_ncu_dir)
    tmp_ext = tempfile.mkdtemp(prefix="torch_ext_")
    env["TORCH_EXTENSIONS_DIR"] = tmp_ext

    tag = label or f"{target_name}/{scenario_name}"
    print(f"  [ncu] profiling {tag}...", flush=True)

    try:
        proc = subprocess.run(
            cmd, env=env, text=True, capture_output=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"  WARNING: NCU timed out after 300s for {tag}")
        return None

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        if "Permission" in stderr or "requires root" in stderr.lower():
            print("  WARNING: NCU requires elevated privileges. "
                  "Run with sudo or configure NVIDIA permissions.")
        else:
            print(f"  WARNING: NCU exited with code {proc.returncode} for {tag}")
            if stderr:
                print(f"  NCU stderr: {stderr[:500]}")
        return None

    if not csv_path.exists():
        print(f"  WARNING: NCU did not produce CSV for {tag}")
        return None

    print(f"  [ncu] done: {tag}", flush=True)
    return csv_path


# ---------------------------------------------------------------------------
# CSV parsing (adapted from CudaForge)
# ---------------------------------------------------------------------------
def load_ncu_metrics(
    csv_path: str | Path,
    columns: Sequence[str] | None = None,
    extra_keep: Sequence[str] = ("Kernel Name",),
    kernel_names: Sequence[str] | None = None,
    select: str = "last",
) -> pd.DataFrame:
    """Parse an NCU CSV into a clean numeric DataFrame.

    Parameters
    ----------
    csv_path:
        Path to the NCU ``--csv`` output file.
    columns:
        Metric columns to extract (default: all 23 ``METRIC_COLUMNS``).
    extra_keep:
        Non-metric columns to keep (e.g. ``"Kernel Name"``).
    kernel_names:
        Optional list of kernel name substrings to filter by.
    select:
        Row selection policy when multiple rows match one kernel name:
        ``"first"``, ``"last"`` (default), or ``"max_cycles"``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"NCU CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, comment="=", low_memory=False)

    metric_cols = list(columns) if columns is not None else METRIC_COLUMNS
    keep_cols: list[str] = []
    if extra_keep:
        keep_cols.extend([c for c in extra_keep if c in df.columns])
    keep_cols.extend([c for c in metric_cols if c in df.columns])
    if not keep_cols:
        raise ValueError("No requested columns found in the NCU CSV header.")

    sub = df[keep_cols].copy()

    # Drop the units row (first row often contains %, inst, cycle, etc.)
    if len(sub) > 0:
        first_row_str = sub.iloc[0].astype(str).str.lower()
        unit_tokens = ("%", "inst", "cycle", "block", "register", "register/thread")
        if first_row_str.apply(lambda x: any(tok in x for tok in unit_tokens)).any():
            sub = sub.iloc[1:].reset_index(drop=True)

    # Coerce metric columns to numeric
    metric_in_sub = [c for c in metric_cols if c in sub.columns]
    sub[metric_in_sub] = (
        sub[metric_in_sub]
        .replace({",": "", "%": ""}, regex=True)
        .apply(pd.to_numeric, errors="coerce")
    )

    # Filter by kernel name list
    if kernel_names and "Kernel Name" in sub.columns:
        results = []
        for name in kernel_names:
            matched = sub[
                sub["Kernel Name"].astype(str).str.contains(name, regex=False, na=False)
            ]
            if matched.empty:
                continue
            if len(matched) > 1:
                if select == "first":
                    row = matched.iloc[[0]]
                elif select == "max_cycles" and "sm__cycles_active.avg" in matched.columns:
                    row = matched.sort_values(
                        "sm__cycles_active.avg", ascending=False
                    ).head(1)
                else:  # "last" or fallback
                    row = matched.iloc[[-1]]
            else:
                row = matched
            results.append(row)

        if results:
            sub = pd.concat(results, ignore_index=True)
        else:
            sub = pd.DataFrame(columns=keep_cols)

    return sub


# ---------------------------------------------------------------------------
# DataFrame -> JSON string for LLM prompts (adapted from CudaForge)
# ---------------------------------------------------------------------------
def metrics_to_prompt(
    df: pd.DataFrame,
    key_by: str = "Kernel Name",
    round_digits: int | None = 3,
    compact: bool = False,
) -> str:
    """Convert an NCU metrics DataFrame to a JSON string.

    Output is keyed by kernel name::

        {
          "kernel_name": {"metric1": 1.23, "metric2": 45.6, ...}
        }

    If the key column is absent, returns a list of row dicts.
    """

    def _safe(v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (pd.Timestamp, pd.Timedelta, pd.Interval)):
            return str(v)
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if isinstance(v, float) and round_digits is not None:
            return round(v, round_digits)
        return v

    if df is None or df.empty:
        return "{}"

    cols = list(df.columns)

    # Round numeric columns
    if round_digits is not None:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df = df.copy()
            df[num_cols] = df[num_cols].round(round_digits)

    # No key column -> list of rows
    if key_by not in cols:
        rows = [
            {k: _safe(v) for k, v in rec.items()}
            for rec in df.to_dict(orient="records")
        ]
        return json.dumps(rows, ensure_ascii=False, indent=None if compact else 2)

    value_cols = [c for c in cols if c != key_by]

    data: dict[str, Any] = {}
    for rec in df[[key_by] + value_cols].to_dict(orient="records"):
        k = str(rec.pop(key_by))
        val_obj = {ck: _safe(cv) for ck, cv in rec.items()}
        if k in data:
            if isinstance(data[k], list):
                data[k].append(val_obj)
            else:
                data[k] = [data[k], val_obj]
        else:
            data[k] = val_obj

    return json.dumps(data, ensure_ascii=False, indent=None if compact else 2)


# ---------------------------------------------------------------------------
# High-level: profile both baseline and candidate for one scenario
# ---------------------------------------------------------------------------
def profile_scenario(
    target_name: str,
    scenario: Scenario,
    target: BenchTarget,
    candidate_path: str,
    num_launches: int = 20,
) -> dict[str, str] | None:
    """Profile both baseline and candidate for a single scenario.

    Parameters
    ----------
    target_name:
        Operator name.
    scenario:
        A ``Scenario`` from the ``InputRegistry`` (has ``.name`` and
        ``.init_args``).
    target:
        ``BenchTarget`` with ``.module_path`` and ``.target_cls``.
    candidate_path:
        File path to the candidate ``.py``.
    num_launches:
        Number of forward() calls NCU will capture.

    Returns
    -------
    Dict with ``baseline_metrics`` and ``candidate_metrics`` JSON strings,
    or ``None`` if NCU is unavailable or both profiles failed.
    """
    if find_ncu_binary() is None:
        return None

    class_name = target.target_cls.__name__

    # Profile baseline
    baseline_csv = profile_kernel(
        target_name=target_name,
        scenario_name=scenario.name,
        class_name=class_name,
        module_source="baseline",
        module_path=target.module_path,
        init_args=scenario.init_args,
        num_launches=num_launches,
        label="baseline",
    )

    # Profile candidate
    candidate_csv = profile_kernel(
        target_name=target_name,
        scenario_name=scenario.name,
        class_name=class_name,
        module_source=candidate_path,
        module_path=target.module_path,
        init_args=scenario.init_args,
        num_launches=num_launches,
        label="candidate",
    )

    if baseline_csv is None and candidate_csv is None:
        return None

    baseline_json = "{}"
    candidate_json = "{}"

    if baseline_csv is not None:
        try:
            baseline_df = load_ncu_metrics(baseline_csv)
            baseline_json = metrics_to_prompt(baseline_df)
        except Exception as e:
            print(f"  WARNING: Failed to parse baseline NCU CSV: {e}")

    if candidate_csv is not None:
        try:
            candidate_df = load_ncu_metrics(candidate_csv)
            candidate_json = metrics_to_prompt(candidate_df)
        except Exception as e:
            print(f"  WARNING: Failed to parse candidate NCU CSV: {e}")

    return {
        "baseline_metrics": baseline_json,
        "candidate_metrics": candidate_json,
    }
