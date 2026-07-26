#!/usr/bin/env python3
"""Package every RUNNABLE kb operator as an AKO4X task (definition + workloads +
expert blob), verifying each seed against the real grader before it ships.

Generalizes ``ako4x_overlay/make_dataset.py`` (which hard-coded ``rms_norm``) to
the whole census. Output layout — exactly what ``spawn.py`` globs, plus a
``blobs/`` sidecar that ``run_campaigns.sh`` copies into each child root:

    <out>/definitions/kb/kb_<op>.json     # {name, op_type, description, axes,
                                          #  inputs, outputs, reference}
    <out>/workloads/kb/kb_<op>.jsonl      # one line per registry scenario
    <out>/blobs/kb_<op>.json              # pack()-format expert baseline
    <out>/PACKAGING_REPORT.{md,json}      # packaged / skipped + reason

Sources of truth
----------------
* ``docs/agent_eval/census_v2.json`` — which ops are RUNNABLE (verdict field).
* ``bench/kernels/benchmark_scenarios/small/shape_registry.yaml`` — the
  scenarios. A workload's ``uuid`` IS the kb scenario name, so
  ``--scenarios <uuid,...>`` on the entrypoint selects exactly that workload.
* ``tasks/reference/L{1,2}/<op>.py`` — the seed the agent starts from. These are
  self-contained pure-torch impls (helpers are already inlined upstream). A seed
  never imports ``tasks.baseline`` — delegating would hand the agent the answer.

Seed contract (why a seed can be rejected)
------------------------------------------
The grader instantiates baseline and candidate from the SAME traced
``init_args``, copies the baseline's ``state_dict`` into the candidate with a
strict missing/unexpected check, then compares outputs AND mutated inputs. So a
seed must declare the baseline's class name, accept the baseline's ``__init__``
keywords, expose the same parameter names, and match the baseline's forward
signature. Every seed is run through the real grader (``agent_entrypoint.py``,
subprocess, one free GPU) over ALL of the op's scenarios; anything short of
all-PASSED is skipped and reported rather than shipped.

Usage
-----
    /raid/user_data/olu/venv/bin/python tools/agent_eval/package_tasks.py \
        --gpus 3,4,5 --out /raid/user_data/olu/agents/kb-trace

    ... --ops gelu,flashinfer_decode        # subset
    ... --no-verify                         # dev aid; report is marked UNVERIFIED
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = Path("/home/olu/kb_nano")
DEFAULT_PYTHON = "/raid/user_data/olu/venv/bin/python"
DEFAULT_OUT = Path("/raid/user_data/olu/agents/kb-trace")
DEFAULT_WORKDIR = Path("/raid/user_data/olu/scratch/agent_eval_pilot/pkg/work")

OP_TYPE = "kb"
DEF_PREFIX = "kb_"
MAX_AXIS_VALUES = 24  # longer value lists are summarized as min/max/count


# ===========================================================================
# Seed patches
# ===========================================================================
#
# A patch is appended to the reference source when the reference file does not
# define the class the grader actually benchmarks. The grader's target is
# ``_find_module_class(baseline_module)`` = the LAST nn.Module defined in the
# baseline file, which is not always the class the reference file mirrors.
#
# Patches must be pure torch and trivially derivable from the baseline's own
# source; they must NOT import tasks.baseline (that would delegate).

SEED_PATCHES: dict[str, str] = {
    # (empty) All three historical patches were retired on 2026-07-26:
    # softmax    -- superseded by the registry class pin (op now grades the
    #               true Softmax class, which the raw reference mirrors);
    # flash_attn_varlen / flux_attention -- their reference files were
    #               fixed in place (keyword-only call / FP32RMSNorm alias).
    # The mechanism stays for future cases; raw-reference verification of
    # all three post-retirement is recorded in the E3 stream reports.
}


# ===========================================================================
# Static helpers (no torch, no GPU)
# ===========================================================================

def runnable_ops(census_path: Path) -> list[str]:
    census = json.loads(census_path.read_text())
    return sorted(op for op, rec in census.items() if rec.get("verdict") == "RUNNABLE")


def _level_path(repo: Path, kind: str, op: str) -> tuple[Path | None, int | None]:
    for level in (1, 2, 3, 4):
        path = repo / "tasks" / kind / f"L{level}" / f"{op}.py"
        if path.is_file():
            return path, level
    return None, None


def reference_path(repo: Path, op: str) -> tuple[Path | None, int | None]:
    return _level_path(repo, "reference", op)


def baseline_path(repo: Path, op: str) -> tuple[Path | None, int | None]:
    return _level_path(repo, "baseline", op)


def _last_class_name(path: Path) -> str | None:
    """Static stand-in for ``_find_module_class`` (last top-level class)."""
    names = [n.name for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef)]
    return names[-1] if names else None


def _seed_body(repo: Path, op: str) -> tuple[str | None, Path | None, str]:
    """``(body, reference_path, note)`` — everything in the seed except the
    generated header. Single source of truth for both build_seed() and the
    content hash that ``--reuse-verify`` keys on."""
    ref, _ = reference_path(repo, op)
    if ref is None:
        return None, None, ""
    body = ref.read_text()
    note = f"tasks/reference/{ref.parent.name}/{ref.name}"
    patch = SEED_PATCHES.get(op)
    if patch:
        body = body.rstrip("\n") + "\n" + patch
        note += f" + SEED_PATCHES[{op}]"
    return body, ref, note


def seed_body_sha(repo: Path, op: str) -> str | None:
    body, _ref, _note = _seed_body(repo, op)
    return None if body is None else hashlib.sha256(body.encode()).hexdigest()


def build_seed(repo: Path, op: str, contract: dict | None) -> tuple[str | None, str]:
    """Return ``(seed_source, note)``. ``seed_source is None`` -> not derivable."""
    ref, _ = reference_path(repo, op)
    if ref is None:
        base, level = baseline_path(repo, op)
        where = f"tasks/baseline/L{level}/{op}.py" if base else "(no baseline either)"
        return None, (
            f"no tasks/reference/L*/{op}.py; the baseline ({where}) is a composite "
            f"that wires sub-modules, so a correct pure-torch seed is not trivially "
            f"derivable (it would have to reproduce the whole sub-module tree, "
            f"including state_dict key paths)"
        )
    body, ref, note = _seed_body(repo, op)
    return _seed_header(op, ref, contract) + body, note


def _seed_header(op: str, ref: Path, contract: dict | None) -> str:
    """Comment block prepended to the seed (keeps the file's own docstring)."""
    lines = [
        f"# Seed kernel for the kb_{op} task — self-contained pure PyTorch.",
        f"# Source: {ref.relative_to(ref.parents[3])} (kb semantic reference).",
        "#",
        "# The harness constructs baseline and candidate from the SAME traced",
        "# init_args, copies the baseline's state_dict into the candidate with a",
        "# strict missing/unexpected check, then compares returned outputs AND",
        "# mutated inputs. Keep the frozen contract below or the scenario fails",
        "# with RUNTIME_ERROR before any numerics run.",
    ]
    if contract:
        lines += [
            "#",
            f"#   class {contract['class_name']}(nn.Module)",
            f"#     __init__{contract['init_signature']}",
            f"#     forward{contract['forward_signature']}",
            "#   state_dict keys: %s" % (
                ", ".join(contract["state_dict_keys"]) or "(none — no parameters)"),
        ]
    return "\n".join(lines) + "\n\n"


# ===========================================================================
# Registry -> axes / workloads / definition
# ===========================================================================

_TOKENS_IN_NAME = re.compile(r"(?:^|/)tokens-(\d+)(?:/|$)")
_TOK_SUFFIX = re.compile(r"-tok(\d+)$")


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (int, float, bool, str))


def _token_count(scenario: dict) -> int:
    """Grouping value. The registry names most scenarios ``tokens-<N>/<hash>``;
    a few name them ``<layer>/<shape>-tok<N>``. When neither encodes it, fall
    back to the leading dim of the first tensor input (alphabetical)."""
    name = scenario["name"]
    m = _TOKENS_IN_NAME.search(name) or _TOK_SUFFIX.search(name)
    if m:
        return int(m.group(1))
    for _key, spec in sorted(scenario.get("inputs", {}).items()):
        if isinstance(spec, dict) and spec.get("shape"):
            return int(spec["shape"][0])
    return 1


def scenario_axis_row(scenario: dict) -> dict:
    """Flat, JSON-safe axes for one scenario.

    Tensor inputs flatten to ``<arg>_dim<i>`` / ``<arg>_dtype`` — the same
    naming ``agent_entrypoint._scenario_axes`` emits into results.json, so the
    dataset side and the results side read alike.
    """
    row: dict = {"tokens": _token_count(scenario)}
    for key, val in sorted(scenario.get("init_args", {}).items()):
        if key == "training":       # always False; a knob the graded module drops
            continue
        if _is_scalar(val):
            row[key] = val
    for key, spec in sorted(scenario.get("inputs", {}).items()):
        if isinstance(spec, dict) and "shape" in spec:
            for i, dim in enumerate(spec["shape"]):
                row[f"{key}_dim{i}"] = int(dim)
            if spec.get("dtype") is not None:
                row[f"{key}_dtype"] = str(spec["dtype"])
        elif _is_scalar(spec):
            row[key if key not in row else f"input_{key}"] = spec
    return row


def _sorted_values(values: list):
    """Numbers numerically, then strings, then everything else — NOT by their
    JSON text (which would order 1024 < 128 < 13692)."""
    def key(v):
        if isinstance(v, bool):
            return (1, "", int(v))
        if isinstance(v, (int, float)):
            return (0, "", v)
        if isinstance(v, str):
            return (2, v, 0)
        return (3, json.dumps(v, sort_keys=True), 0)
    return sorted(values, key=key)


def _axis_description(name: str, scenarios: list[dict]) -> str:
    if name == "tokens":
        return ("scenario size label from the kb registry scenario name "
                "(the grouping axis for scoring)")
    m = re.match(r"^(.*)_dim(\d+)$", name)
    if m:
        return f"dim {m.group(2)} of input tensor `{m.group(1)}`"
    if name.endswith("_dtype"):
        return f"dtype of input tensor `{name[:-len('_dtype')]}`"
    in_init = any(name in s.get("init_args", {}) for s in scenarios)
    if in_init:
        return f"traced `__init__` argument `{name}`"
    return f"traced forward keyword `{name}`"


def build_axes(scenarios: list[dict], rows: list[dict]) -> dict:
    """Definition ``axes``. Insertion order puts ``tokens`` first, so
    ``bench_utils.find_group_axis`` (first ``type == "var"`` axis) groups by
    token count whenever it actually varies, and falls through to the next
    varying axis when it does not."""
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)

    axes: dict = {}
    for name in ordered:
        present = [row[name] for row in rows if name in row]
        missing = len(present) != len(rows)
        uniq = {json.dumps(v, sort_keys=True) for v in present}
        values = _sorted_values([json.loads(v) for v in uniq])
        desc = _axis_description(name, scenarios)
        if len(values) == 1 and not missing:
            axes[name] = {"type": "const", "value": values[0], "description": desc}
            continue
        entry = {"type": "var", "description": desc}
        if missing:
            entry["description"] += " (absent in some scenarios)"
        if len(values) <= MAX_AXIS_VALUES:
            entry["values"] = values
        else:
            nums = [v for v in values if isinstance(v, (int, float))
                    and not isinstance(v, bool)]
            entry["n_values"] = len(values)
            if nums:
                entry["min"] = min(nums)
                entry["max"] = max(nums)
        axes[name] = entry
    return axes


def build_inputs(scenarios: list[dict], axes: dict) -> dict:
    """Definition ``inputs``: tensor specs with symbolic (axis-named) dims plus
    the scalar forward keywords the scenarios pass."""
    keys: list[str] = []
    for s in scenarios:
        for k in sorted(s.get("inputs", {})):
            if k not in keys:
                keys.append(k)

    out: dict = {}
    for key in keys:
        specs = [s["inputs"][key] for s in scenarios if key in s.get("inputs", {})]
        optional = len(specs) != len(scenarios)
        tensor = [sp for sp in specs if isinstance(sp, dict) and "shape" in sp]
        if tensor:
            ranks = sorted({len(sp["shape"]) for sp in tensor})
            dtypes = sorted({str(sp.get("dtype")) for sp in tensor})
            entry = {
                "dtype": dtypes[0] if len(dtypes) == 1 else "|".join(dtypes),
                "description": f"tensor argument `{key}`",
            }
            if len(ranks) == 1:
                shape: list = []
                for i in range(ranks[0]):
                    dims = {sp["shape"][i] for sp in tensor}
                    shape.append(int(next(iter(dims))) if len(dims) == 1
                                 else f"{key}_dim{i}")
                entry["shape"] = shape
            else:
                # Rank itself varies across scenarios (e.g. gelu is traced on
                # 2-D … 5-D activations). A single symbolic shape would be a
                # lie; the per-workload dims live in the <key>_dim<i> axes.
                entry["rank"] = ranks
                entry["shape"] = (f"variable rank {ranks}; per-workload dims are "
                                  f"the `{key}_dim<i>` axes")
            if optional:
                entry["optional"] = True
            out[key] = entry
        else:
            vals = {json.dumps(sp, sort_keys=True) for sp in specs}
            entry = {
                "kind": "scalar",
                "description": f"non-tensor forward keyword `{key}`",
                "values": _sorted_values([json.loads(v) for v in vals])[:MAX_AXIS_VALUES],
            }
            if optional:
                entry["optional"] = True
            out[key] = entry
    return out


def build_outputs(contract: dict | None) -> dict:
    """Definition ``outputs``, from the baseline's observed forward result."""
    spec = (contract or {}).get("output_spec")
    if not spec:
        return {"out": {
            "description": "return value of the baseline's forward (structure not "
                           "probed at packaging time); the harness compares the "
                           "whole returned tree AND any mutated inputs",
        }}
    probe = (contract or {}).get("probe_scenario", "the first scenario")
    out: dict = {}
    for item in spec:
        name = item["name"]
        entry = {"description": f"{item['description']}; shape/dtype observed on "
                                f"scenario `{probe}` — other scenarios differ, see "
                                f"the axes"}
        if item.get("shape") is not None:
            entry["shape"] = item["shape"]
        if item.get("dtype") is not None:
            entry["dtype"] = item["dtype"]
        out[name] = entry
    return out


def build_description(op: str, scenarios: list[dict], axes: dict,
                      contract: dict | None, seed_note: str) -> str:
    group_axis = next((n for n, a in axes.items() if a["type"] == "var"), None)
    n_var = sum(1 for a in axes.values() if a["type"] == "var")
    dtypes = sorted({str(sp.get("dtype")) for s in scenarios
                     for sp in s.get("inputs", {}).values()
                     if isinstance(sp, dict) and sp.get("dtype") is not None})
    lines = [
        f"kb-nano/fastkernels operator `{op}`"
        + (f" (baseline: tasks/baseline/L{contract['level']}/{op}.py, class "
           f"{contract['class_name']})." if contract else "."),
        "",
        f"{len(scenarios)} scenario(s) taken verbatim from the kb shape registry "
        f"(operator '{op}', small suite). Each workload uuid IS the registry "
        f"scenario name. {n_var} axis/axes vary"
        + (f"; results are grouped by `{group_axis}`." if group_axis
           else " (single-scenario operator).")
        + (f" Tensor dtypes in play: {', '.join(dtypes)}." if dtypes else ""),
    ]
    if contract:
        lines += [
            "",
            "Interface contract (frozen — the harness copies the baseline's "
            "state_dict into your module and calls it with the traced kwargs):",
            f"  class {contract['class_name']}(nn.Module)",
            f"    __init__{contract['init_signature']}",
            f"    forward{contract['forward_signature']}",
            "  state_dict keys: %s" % (
                ", ".join(contract["state_dict_keys"]) or "(none — no parameters)"),
            "",
            "Renaming/adding/removing parameters fails the scenario with "
            "RUNTIME_ERROR (weight_transfer_incomplete) before any numerics run. "
            "Constructor keywords the class does not declare are filtered out, so "
            "accepting **kwargs is safe but not required. Mutated inputs are "
            "compared as well as returned outputs: an in-place baseline path must "
            "stay in-place.",
        ]
    lines += [
        "",
        f"Seed (`reference` field, copied to solution/kernel.py at spawn): "
        f"{seed_note}. It PASSES every scenario under the grader — it is a "
        f"correct-but-slow starting point, not a stub.",
    ]
    return "\n".join(lines)


def build_definition(op: str, scenarios: list[dict], rows: list[dict],
                     contract: dict | None, seed_src: str, seed_note: str) -> dict:
    axes = build_axes(scenarios, rows)
    return {
        "name": DEF_PREFIX + op,
        "op_type": OP_TYPE,
        "description": build_description(op, scenarios, axes, contract, seed_note),
        "axes": axes,
        "inputs": build_inputs(scenarios, axes),
        "outputs": build_outputs(contract),
        "reference": seed_src,
    }


def build_workloads(op: str, scenarios: list[dict], rows: list[dict]) -> list[dict]:
    return [
        {
            "definition": DEF_PREFIX + op,
            "solution": None,
            "workload": {"uuid": s["name"], "axes": row, "inputs": {}},
            "evaluation": None,
        }
        for s, row in zip(scenarios, rows)
    ]


# ===========================================================================
# Expert blob
# ===========================================================================

EXPERT_TEMPLATE = '''"""Expert baseline for {defname} — the kb production baseline itself.

INFRA, not an agent solution: bench_utils profiles this at the child's root to
produce the score denominator. The delegation ban that applies to
solution/kernel.py does NOT apply here — measuring the incumbent kb kernel is
the whole point.

Imported through the `fastkernels` package name (not a bare `tasks.baseline`
path) because the entrypoint binds `fastkernels` in sys.modules before importing
the candidate: this resolves to the very module object already loaded as the
baseline, so expert and correctness oracle are the same class — no second copy,
no second JIT build of the CUDA extension.
"""

from fastkernels.tasks.baseline.L{level}.{op} import {cls} as {cls}  # noqa: F401
'''


def build_expert_blob(adapter, op: str, level: int, cls: str) -> str:
    src = EXPERT_TEMPLATE.format(defname=DEF_PREFIX + op, level=level, op=op, cls=cls)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "kernel.py").write_text(src)
        return adapter.pack(
            tmp,
            {"language": "python", "entry_point": "kernel.py::run",
             "destination_passing_style": False},
            name=f"{DEF_PREFIX}{op}-expert-baseline",
            definition=DEF_PREFIX + op,
            author="baseline",
        )


# ===========================================================================
# Worker: introspect the baseline, then grade the seed
# ===========================================================================

def introspect(repo: Path, op: str, scenario_name: str) -> dict:
    """Authoritative contract for `op`, read off the live baseline.

    Reuses agent_entrypoint as a library (read-only) so the class resolution,
    init-arg filtering and input preparation are IDENTICAL to the grader's.
    """
    import inspect as _inspect

    sys.path.insert(0, str(HERE))
    os.environ.setdefault("FASTKERNELS_TREE", str(repo))
    import agent_entrypoint as AE

    AE._bootstrap_fastkernels()
    AE._ensure_ninja_on_path()
    target = AE._resolve_target(op)
    cls = target.target_cls

    info = {
        "class_name": cls.__name__,
        "level": target.level,
        "module_path": target.module_path,
        "probe_scenario": scenario_name,
        "init_signature": str(_inspect.signature(cls.__init__)).replace("self, ", "", 1)
                          if cls.__init__ is not object.__init__ else "(self)",
        "forward_signature": str(_inspect.signature(cls.forward)).replace("self, ", "", 1),
        "state_dict_keys": [],
        "output_spec": None,
    }

    import torch  # noqa: F401
    from fastkernels.bench.kernels import runner as R
    from fastkernels.bench.kernels.scenario_registry import InputRegistry

    registry = InputRegistry()
    scenario = next(s for s in registry.scenarios(op) if s.name == scenario_name)
    inputs = registry.get_inputs(op, scenario.name, device="cuda")
    inputs = AE._prepare_inputs_for_target(op, inputs, "cuda", scenario.init_args)
    dtype = R._first_floating_dtype(inputs)
    module = AE._instantiate_module(cls, scenario.init_args, "cuda", dtype=dtype)
    AE._repair_degenerate_parameters(module, op)
    info["state_dict_keys"] = sorted(module.state_dict())

    out = R._run_forward_once(module, R._clone_inputs(inputs))
    info["output_spec"] = _describe_output(out)
    return info


def _describe_output(out) -> list[dict]:
    import torch

    def one(name, val, desc):
        if isinstance(val, torch.Tensor):
            return {"name": name, "shape": list(val.shape),
                    "dtype": str(val.dtype).replace("torch.", ""), "description": desc}
        return {"name": name, "shape": None, "dtype": None,
                "description": f"{desc} (type {type(val).__name__})"}

    if isinstance(out, (tuple, list)):
        return [one(f"out{i}", v, f"element {i} of the returned tuple")
                for i, v in enumerate(out)]
    if isinstance(out, dict):
        return [one(k, v, f"key '{k}' of the returned mapping") for k, v in out.items()]
    return [one("out", out, "the returned tensor")]


def grade(python: str, entrypoint: Path, repo: Path, op: str, seed_file: Path,
          timeout: float) -> dict:
    """Run the REAL grader on the seed over every scenario of `op`."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["FASTKERNELS_TREE"] = str(repo)
    cmd = [python, str(entrypoint), "--op", op, "--candidate", str(seed_file)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env, cwd=str(repo))
    except subprocess.TimeoutExpired:
        return {"verdict": "TIMEOUT", "seconds": round(time.time() - t0, 1),
                "cmd": " ".join(cmd),
                "detail": f"grader exceeded {timeout:g}s"}
    seconds = round(time.time() - t0, 1)
    if proc.returncode != 0:
        return {"verdict": "GRADER_ERROR", "seconds": seconds, "cmd": " ".join(cmd),
                "detail": (proc.stderr or "")[-1500:]}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"verdict": "BAD_OUTPUT", "seconds": seconds, "cmd": " ".join(cmd),
                "detail": (proc.stdout or "")[-1500:]}
    results = payload.get(DEF_PREFIX + op, payload.get("results", {}))
    if isinstance(results, dict) and DEF_PREFIX + op in results:
        results = results[DEF_PREFIX + op]
    tally: dict[str, int] = {}
    failures = []
    for name, entry in results.items():
        st = entry.get("status", "?")
        tally[st] = tally.get(st, 0) + 1
        if st != "PASSED" and len(failures) < 4:
            failures.append({"scenario": name, "status": st,
                             "error_log": (entry.get("error_log") or "")[:600]})
    verdict = "SEED_PASS" if tally.get("PASSED", 0) == len(results) and results \
        else "SEED_FAIL"
    return {"verdict": verdict, "seconds": seconds, "cmd": " ".join(cmd),
            "n_scenarios": len(results), "tally": tally, "failures": failures}


def worker_main(args) -> int:
    """One op, one GPU. Writes <workdir>/verify/<op>.json and exits 0."""
    repo, op = Path(args.repo), args.op
    out_path = Path(args.workdir) / "verify" / f"{op}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict = {"op": op, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?")}
    try:
        import yaml
        scenarios = yaml.safe_load(Path(args.registry).read_text())[op]["scenarios"]
        record["n_scenarios"] = len(scenarios)

        contract = None
        try:
            contract = introspect(repo, op, scenarios[0]["name"])
        except Exception as exc:                                  # noqa: BLE001
            record["introspect_error"] = f"{type(exc).__name__}: {exc}"
            static, level = baseline_path(repo, op)
            if static is not None:
                contract = {"class_name": _last_class_name(static), "level": level,
                            "module_path": f"tasks.baseline.L{level}.{op}",
                            "init_signature": "(...)", "forward_signature": "(...)",
                            "state_dict_keys": [], "output_spec": None,
                            "static_fallback": True}
        record["contract"] = contract

        seed_src, seed_note = build_seed(repo, op, contract)
        record["seed_note"] = seed_note
        if seed_src is None:
            record["verdict"] = "NO_SEED"
            record["detail"] = seed_note
            out_path.write_text(json.dumps(record, indent=2, default=str))
            return 0
        seed_file = Path(args.workdir) / "seeds" / f"kb_{op}.py"
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        seed_file.write_text(seed_src)
        record["seed_file"] = str(seed_file)
        record["seed_sha256"] = hashlib.sha256(seed_src.encode()).hexdigest()
        record["seed_body_sha256"] = seed_body_sha(repo, op)

        record.update(grade(args.python, Path(args.entrypoint), repo, op,
                            seed_file, args.timeout))
    except Exception:                                             # noqa: BLE001
        record["verdict"] = "WORKER_CRASH"
        record["detail"] = traceback.format_exc()[-2000:]
    out_path.write_text(json.dumps(record, indent=2, default=str))
    return 0


# ===========================================================================
# Driver
# ===========================================================================

def _reusable(repo: Path, op: str, rec: dict) -> bool:
    """True when `rec`'s verdict still describes the seed we would ship today.

    Rebuilds the seed from the record's own contract and compares against the
    hash the worker stored for the file it actually graded. Exact: a SEED_PATCH,
    a tasks/reference/ edit, or a header/contract change all break the match and
    force a re-verify.
    """
    if not rec.get("verdict"):
        return False
    src, _note = build_seed(repo, op, rec.get("contract"))
    if src is None:
        return rec["verdict"] == "NO_SEED"
    want = rec.get("seed_sha256")
    return bool(want) and want == hashlib.sha256(src.encode()).hexdigest()


def run_verification(ops: list[str], args) -> dict[str, dict]:
    """One subprocess per op, dynamically scheduled over the GPU slots."""
    slots: "queue.Queue[str]" = queue.Queue()
    for gpu in args.gpus.split(","):
        for _ in range(args.jobs_per_gpu):
            slots.put(gpu.strip())
    n_slots = slots.qsize()
    if not n_slots:
        raise SystemExit("--gpus resolved to no GPU slots")

    done = {"n": 0}
    lock = threading.Lock()
    verify_dir = Path(args.workdir) / "verify"

    def one(op: str) -> dict:
        # Content-addressed reuse: an existing record is only trusted when the
        # seed BODY it was produced from still hashes the same, so adding a
        # SEED_PATCH or editing tasks/reference/ forces a re-verify.
        if args.reuse_verify:
            cached = verify_dir / f"{op}.json"
            if cached.is_file():
                rec = json.loads(cached.read_text())
                if _reusable(Path(args.repo), op, rec):
                    with lock:
                        done["n"] += 1
                        print(f"[{done['n']}/{len(ops)}] {op:32s} "
                              f"{rec['verdict']:14s} (reused)", flush=True)
                    return rec

        gpu = slots.get()
        try:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = args.repo
            cmd = [args.python, str(Path(__file__).resolve()), "--worker-op", op,
                   "--repo", args.repo, "--registry", args.registry,
                   "--workdir", args.workdir, "--python", args.python,
                   "--entrypoint", args.entrypoint, "--timeout", str(args.timeout)]
            log = verify_dir / f"{op}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as fh:
                subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=args.timeout + 600)
        except Exception as exc:                                  # noqa: BLE001
            (verify_dir / f"{op}.json").write_text(json.dumps(
                {"op": op, "verdict": "LAUNCH_FAIL", "detail": repr(exc)}, indent=2))
        finally:
            slots.put(gpu)
        path = verify_dir / f"{op}.json"
        rec = json.loads(path.read_text()) if path.is_file() else \
            {"op": op, "verdict": "NO_RECORD", "detail": "worker wrote no record"}
        with lock:
            done["n"] += 1
            print(f"[{done['n']}/{len(ops)}] {op:32s} {rec.get('verdict','?'):14s} "
                  f"gpu={rec.get('gpu','?')} {rec.get('seconds','?')}s", flush=True)
        return rec

    print(f"verifying {len(ops)} op(s) on {n_slots} slot(s) "
          f"(gpus {args.gpus}, {args.jobs_per_gpu}/gpu)", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_slots) as pool:
        records = list(pool.map(one, ops))
    return {r["op"]: r for r in records}


def unverified_records(ops: list[str], args) -> dict[str, dict]:
    """--no-verify path: static contract only, nothing graded."""
    out = {}
    repo = Path(args.repo)
    for op in ops:
        base, level = baseline_path(repo, op)
        contract = None
        if base is not None:
            contract = {"class_name": _last_class_name(base), "level": level,
                        "module_path": f"tasks.baseline.L{level}.{op}",
                        "init_signature": "(...)", "forward_signature": "(...)",
                        "state_dict_keys": [], "output_spec": None,
                        "static_fallback": True}
        seed_src, note = build_seed(repo, op, contract)
        out[op] = {"op": op, "contract": contract, "seed_note": note,
                   "verdict": "UNVERIFIED" if seed_src else "NO_SEED",
                   "detail": note if seed_src is None else ""}
    return out


SKIP_REASONS = {
    "NO_SEED": "no reference impl; baseline is a composite, seed not trivially derivable",
    "GRADER_ERROR": "grader exited nonzero on the seed",
    "TIMEOUT": "grader exceeded the per-op timeout on the seed",
    "BAD_OUTPUT": "grader produced no parseable JSON",
    "WORKER_CRASH": "packaging worker crashed",
    "LAUNCH_FAIL": "could not launch the packaging worker",
    "NO_RECORD": "worker produced no record",
}


def skip_reason(rec: dict) -> str:
    """Finer-grained reason than the raw verdict, for the report's table."""
    verdict = rec.get("verdict")
    if verdict != "SEED_FAIL":
        return SKIP_REASONS.get(verdict, verdict or "?")
    tally = rec.get("tally") or {}
    fails = rec.get("failures") or []
    nonfinite = [f for f in fails
                 if "baseline_output_nonfinite" in (f.get("error_log") or "")]
    # Only the *whole op* being unmeasurable earns the fixture verdict; a single
    # non-finite scenario alongside real reference errors must not mask them.
    if fails and len(nonfinite) == len(fails) and set(tally) <= {"RUNTIME_ERROR"}:
        return ("degenerate fixture: the BASELINE's own output was non-finite, so "
                "nothing about the candidate was measured. These ops are FLAKY, "
                "not uniformly broken — their baseline allocates parameters with "
                "torch.empty and the grader only repairs values that are already "
                "NaN/Inf/zero, so finite allocator garbage passes through and can "
                "overflow. Measured on yolov10_c2f: 3 PASSED / 2 non-finite over 5 "
                "identical --baseline-identity runs")
    has_rt = tally.get("RUNTIME_ERROR", 0) > 0
    has_num = tally.get("INCORRECT_NUMERICAL", 0) > 0
    if has_rt and has_num:
        base = "reference impl raises on some scenarios and disagrees numerically on others"
    elif has_rt:
        base = "reference impl raises on the traced scenarios"
    elif has_num:
        base = "reference impl disagrees with the baseline beyond the dtype tolerance"
    else:
        base = "seed did not pass every scenario under the grader"
    if nonfinite:
        base += " (plus >=1 scenario with a non-finite BASELINE output — that one is a fixture fault)"
    return base


def skip_detail(rec: dict) -> str:
    if rec.get("detail"):
        return rec["detail"]
    tally = rec.get("tally") or {}
    head = ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
    fails = rec.get("failures") or []
    if not fails:
        return head
    f = fails[0]
    log = " ".join((f.get("error_log") or "").split())[:240]
    return f"{head}; first failing scenario `{f['scenario']}` -> {f['status']}: {log}"


def emit(op: str, scenarios: list[dict], record: dict, adapter, out: Path) -> dict:
    rows = [scenario_axis_row(s) for s in scenarios]
    contract = record.get("contract")
    seed_src, seed_note = build_seed(Path(record["repo"]), op, contract)
    # What ships must be byte-identical to what the grader passed.
    graded_sha = record.get("seed_sha256")
    if graded_sha and graded_sha != hashlib.sha256(seed_src.encode()).hexdigest():
        raise RuntimeError(
            f"{op}: rebuilt seed differs from the graded seed "
            f"({record.get('seed_file')}); refusing to ship an unverified seed")
    definition = build_definition(op, scenarios, rows, contract, seed_src, seed_note)
    workloads = build_workloads(op, scenarios, rows)

    def_path = out / "definitions" / OP_TYPE / f"{DEF_PREFIX}{op}.json"
    wl_path = out / "workloads" / OP_TYPE / f"{DEF_PREFIX}{op}.jsonl"
    blob_path = out / "blobs" / f"{DEF_PREFIX}{op}.json"
    for p in (def_path, wl_path, blob_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    def_path.write_text(json.dumps(definition, indent=2) + "\n")
    wl_path.write_text("".join(json.dumps(w) + "\n" for w in workloads))
    blob_path.write_text(build_expert_blob(adapter, op, contract["level"],
                                           contract["class_name"]))
    return {"definition": str(def_path), "workloads": str(wl_path),
            "blob": str(blob_path), "n_workloads": len(workloads),
            "group_axis": next((n for n, a in definition["axes"].items()
                                if a["type"] == "var"), None)}


def write_report(out: Path, packaged: list[dict], skipped: list[dict],
                 args, started: str) -> None:
    payload = {"generated": started, "out": str(out), "verified": not args.no_verify,
               "grader": args.entrypoint, "python": args.python, "repo": args.repo,
               "gpus": args.gpus, "packaged": packaged, "skipped": skipped}
    (out / "PACKAGING_REPORT.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# kb-trace packaging report",
        "",
        f"- generated: {started}",
        f"- dataset root: `{out}`",
        f"- seed verification: "
        + ("**skipped (--no-verify)** — seeds are UNPROVEN" if args.no_verify else
           f"every seed run through `{args.entrypoint}` over ALL of its scenarios "
           f"on GPUs {args.gpus}"),
        f"- packaged: **{len(packaged)}**   skipped: **{len(skipped)}**",
        "",
        "## Packaged",
        "",
        "| op | scenarios | group axis | seed | grader |",
        "|---|---|---|---|---|",
    ]
    for p in sorted(packaged, key=lambda r: r["op"]):
        tally = p.get("tally") or {}
        grade_txt = (", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
                     or "unverified")
        lines.append(f"| `{p['op']}` | {p['n_workloads']} | "
                     f"`{p.get('group_axis') or '(none)'}` | {p['seed_note']} | "
                     f"{grade_txt} |")
    lines += ["", "## Skipped", "",
              "Every skip below is a *seed* problem or a *fixture* problem, not a "
              "statement that the operator is unbenchmarkable: census_v2 records "
              "the kb baseline itself as RUNNABLE for all of them. For the "
              "reference-impl rows, fixing the named `tasks/reference/` file is "
              "what would unlock the task. The fixture rows need the baseline's "
              "`torch.empty` parameters initialized — those ops are flaky under "
              "the grader regardless of what the candidate does.",
              "",
              "| op | scenarios | reason | evidence |", "|---|---|---|---|"]
    for s in sorted(skipped, key=lambda r: r["op"]):
        detail = (s.get("detail") or "").replace("\n", " ").replace("|", "/")[:300]
        lines.append(f"| `{s['op']}` | {s.get('n_scenarios') or '?'} | "
                     f"{s['reason']} | {detail} |")
    lines.append("")
    (out / "PACKAGING_REPORT.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--census", default="")
    ap.add_argument("--registry", default="")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--entrypoint", default="")
    ap.add_argument("--adapter", default="",
                    help="benchmark_adapter.py providing pack() "
                         "(default: tools/agent_eval/ako4x_overlay/)")
    ap.add_argument("--ops", default="", help="comma-separated subset")
    ap.add_argument("--gpus", default="3,4,5")
    ap.add_argument("--jobs-per-gpu", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=2400.0,
                    help="per-op grader budget, seconds")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--reuse-verify", action="store_true",
                    help="reuse a previous run's per-op verdict when the seed "
                         "body still hashes the same (re-verifies anything a "
                         "SEED_PATCH or a tasks/reference/ edit changed)")
    ap.add_argument("--worker-op", dest="op", default="",
                    help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    args.repo = str(repo)
    if not args.census:
        args.census = str(repo / "docs" / "agent_eval" / "census_v2.json")
    if not args.registry:
        args.registry = str(repo / "bench" / "kernels" / "benchmark_scenarios"
                            / "small" / "shape_registry.yaml")
    if not args.entrypoint:
        args.entrypoint = str(HERE / "agent_entrypoint.py")
    if not args.adapter:
        args.adapter = str(HERE / "ako4x_overlay" / "benchmark_adapter.py")

    if args.op:
        return worker_main(args)

    import yaml
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = Path(args.out)
    ops = ([o.strip() for o in args.ops.split(",") if o.strip()]
           or runnable_ops(Path(args.census)))
    registry = yaml.safe_load(Path(args.registry).read_text())
    missing = [o for o in ops if o not in registry]
    if missing:
        raise SystemExit(f"ops absent from the shape registry: {missing}")

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    records = (unverified_records(ops, args) if args.no_verify
               else run_verification(ops, args))

    sys.path.insert(0, str(Path(args.adapter).parent))
    import benchmark_adapter as adapter

    packaged, skipped = [], []
    for op in ops:
        rec = records.get(op, {"verdict": "NO_RECORD"})
        rec["repo"] = args.repo
        verdict = rec.get("verdict")
        ok = verdict in ("SEED_PASS", "UNVERIFIED")
        contract = rec.get("contract")
        if ok and (not contract or not contract.get("class_name")):
            ok, verdict = False, "WORKER_CRASH"
            rec["detail"] = "no baseline class resolved; cannot build the expert blob"
        if not ok:
            skipped.append({"op": op, "reason": skip_reason(rec),
                            "verdict": verdict, "n_scenarios": rec.get("n_scenarios"),
                            "detail": skip_detail(rec)})
            continue
        info = emit(op, registry[op]["scenarios"], rec, adapter, out)
        info.update({"op": op, "seed_note": rec.get("seed_note", "?"),
                     "tally": rec.get("tally"), "grader_seconds": rec.get("seconds"),
                     "class": contract["class_name"], "level": contract["level"]})
        packaged.append(info)

    write_report(out, packaged, skipped, args, started)
    print(f"\npackaged {len(packaged)} / skipped {len(skipped)}  ->  {out}")
    print(f"report: {out / 'PACKAGING_REPORT.md'}")
    for s in skipped:
        print(f"  SKIP {s['op']:32s} {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
