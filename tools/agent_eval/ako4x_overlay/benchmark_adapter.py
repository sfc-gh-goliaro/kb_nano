"""Benchmark adapter — the single seam between AKO4X and the active benchmark.

**Active benchmark: kb-nano / fastkernels** (the `kb_agent_eval` worktree), reached
through a subprocess CLI, not a Python package. This is the **only** module in the
repo that knows how the benchmark is invoked. Everything else (the runners,
``bench_utils``, ``pack_solution``, the cheat-check) reaches the benchmark through
the **plain-data functions** below — no benchmark types ever cross this boundary.

Why a subprocess and not an import: kb's benchmark runtime lives in a *different
virtualenv* from AKO4X's (kb needs torch 2.9 + vLLM + JIT-built CUDA extensions;
AKO4X's env is its own). ``run`` therefore shells out to
``agent_entrypoint.py`` under the kb interpreter and parses the single JSON object
it prints on stdout. That subprocess is also our per-run isolation — see
``use_isolated_runner`` below.

Porting AKO4X to a different benchmark
--------------------------------------
Reimplement the public functions below so they resolve to your benchmark's
runtime, then rewrite the ``benchmark`` SKILL (``templates/skills/benchmark/``)
and ``templates/benchmark/evaluation.toml``. Nothing else under ``scripts/``
should need to change — ``bench_utils`` and the runners only ever pass/receive
``str`` / ``list[str]`` / ``dict``. The full procedure is in ``docs/porting.md``.

Public surface (plain data in, plain data out — no benchmark types escape here)
-------------------------------------------------------------------------------
Discovery   : ``list_workloads(dataset_path, definition) -> [{"uuid","axes"}, ...]``
Packing     : ``pack(source_dir, build_cfg, *, name, definition, author) -> blob:str``
              ``solution_meta(blob) -> {"name","definition","author"}``
Execution   : ``run(blob, uuids, params, *, dataset_path, capture_logs=False,
              capture_autotune=False) -> normalized_result_dict``
Profiling   : ``profile(...) -> str``  [STUB]  ``list_ncu_options() -> str``  [STUB]
Sanitizer   : ``sanitize(...) -> str``  [STUB]
Cheat-check : ``cheat_check(...) -> dict``  [STUB]
Constants   : ``STATUS_*``; ``MODAL_IMAGE_REGISTRY`` / ``MODAL_PYTHON`` /
              ``MODAL_PACKAGE_PIN`` / ``MODAL_EXTRA_PIN``; ``DATASET_PATH_ENV`` /
              ``LEGACY_DATASET_PATH_ENV``; ``NCU_NVTX_RANGE``

The solution **blob** is porter-defined; here it is a JSON document::

    {"name": str, "definition": str, "author": str,
     "sources": {relative_path: file_content, ...},
     "entry_file": "kernel.py", "language": str}

``pack`` collects ``*.py`` under ``source_dir`` (recursively — helper modules are
allowed and are restored next to the entry file); ``run`` materializes them in a
tempdir and points the entrypoint's ``--candidate`` at ``entry_file``. The tempdir
is prepended to the child process's ``PYTHONPATH`` so ``import my_helper`` inside
``kernel.py`` resolves.

Normalized result dict (``run``'s output; consumed by the benchmark-agnostic
scoring / baseline code in ``bench_utils``)::

    {definition_name: {workload_uuid: {
        "status": <str>,                  # one of STATUS_* below
        "solution": <str>,
        "axes": {<axis>: <value>, ...},
        "latency_ms": <float>,            # present when PASSED
        "reference_latency_ms": <float>,
        "speedup_factor": <float>,
        "max_abs_error": <float|"NaN">,   # present when correctness ran
        "max_rel_error": <float|"NaN">,
        "error_log": <str>,               # present for non-PASSED workloads
        "log": <str>,                     # present with capture_logs
    }}}

The entrypoint already emits this shape (definition name ``kb_<op>``, workload
uuid == the kb scenario name), so ``run`` validates and normalizes rather than
translates.

What ``params`` this port honors
--------------------------------
``bench_utils`` sends a fixed key set: ``warmup_runs``, ``iterations``,
``num_trials``, ``atol``, ``rtol``, ``required_matched_ratio``,
``use_isolated_runner``, ``timeout_seconds``, ``profile_baseline``.

- **``timeout_seconds`` — honored**, as a *per-workload* budget: the subprocess
  covers every requested uuid, so the wall-clock limit is
  ``timeout_seconds * len(uuids)``. On expiry every requested uuid is reported
  ``TIMEOUT``.
- **``atol`` / ``rtol`` / ``required_matched_ratio`` — DELIBERATELY IGNORED.**
  They arrive from the child's ``config.toml``, which the *agent under evaluation*
  can edit. Correctness comes solely from the kb entrypoint's own tolerances
  (``bench/kernels/runner.py`` module constants, per-dtype, ``max_error_ratio
  <= 1.0``), which have no CLI knob. An agent cannot loosen its own gate.
- **``warmup_runs`` / ``iterations`` / ``num_trials`` — ignored.** kb's runner
  pins warmup=10, median-of-100 in-process. A consequence worth knowing: the
  expert-baseline profile and the solution profile are measured with the *same*
  protocol here, unlike FIB where the baseline used a lighter config.
- **``use_isolated_runner`` — ignored** (FIB-internal). Every ``run`` call is
  already its own process; set it ``false`` in ``evaluation.toml`` to avoid
  implying otherwise.
- **``profile_baseline`` — ignored.** The kb entrypoint always measures the kb
  baseline alongside the candidate (it needs it as the correctness oracle), so a
  baseline latency is present in every result.

Statuses this port can produce: ``PASSED`` / ``INCORRECT_NUMERICAL`` /
``RUNTIME_ERROR`` (from the entrypoint, per scenario), ``COMPILE_ERROR`` (the
candidate failed to import — the entrypoint exits 2, and ``run`` fans that out to
every requested uuid), ``TIMEOUT`` (subprocess budget exceeded). Any *other*
nonzero exit is an infrastructure failure and raises — it must not be laundered
into a per-workload status, or a broken environment would read as a bad kernel.
"""

# No module-level benchmark import: this adapter reaches kb through a subprocess,
# so importing it costs nothing and requires neither torch nor the kb tree. Only
# an actual `run` call touches the benchmark. Because no benchmark types cross the
# public surface (everything is str / dict / list), Modal would cloudpickle only
# builtins — though the Modal backend is unsupported here (see MODAL_* below).

import json
import os
import subprocess
import tempfile
from pathlib import Path

# --- Status enum (the active benchmark's per-workload outcome strings) -------
# STATUS_PASSED is load-bearing as the literal "PASSED": bench_utils.compute_score
# filters on the string, not the constant.
STATUS_PASSED = "PASSED"
STATUS_COMPILE_ERROR = "COMPILE_ERROR"
STATUS_INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
STATUS_TIMEOUT = "TIMEOUT"

_VALID_STATUSES = frozenset({
    STATUS_PASSED, STATUS_COMPILE_ERROR, STATUS_INCORRECT_NUMERICAL,
    STATUS_RUNTIME_ERROR, STATUS_TIMEOUT,
})

# --- Dataset discovery -------------------------------------------------------
DATASET_PATH_ENV = "AKO_DATASET_PATH"
LEGACY_DATASET_PATH_ENV = "FIB_DATASET_PATH"

# --- NCU profiling -----------------------------------------------------------
# Stubbed for kb (no NCU agent). Kept non-empty because the profiler-ncu skill
# quotes it in its "No kernels were profiled" diagnosis text.
NCU_NVTX_RANGE = "kb_ncu_profile_unsupported"

# --- Modal image pins --------------------------------------------------------
# UNSUPPORTED BACKEND. kb's runtime is a local venv + JIT-built CUDA extensions on
# a specific host; there is no published Modal image. The constants exist because
# the run_modal* launchers import them at module load. Use --backend local.
MODAL_IMAGE_REGISTRY = ""
MODAL_PYTHON = ""
MODAL_PACKAGE_PIN = ""
MODAL_EXTRA_PIN = ""

# --- kb invocation configuration --------------------------------------------
# Overridable per host via env; the defaults are this pilot's verified paths.
KB_PYTHON_ENV = "KB_EVAL_PYTHON"
KB_REPO_ENV = "KB_EVAL_REPO"
KB_ENTRYPOINT_ENV = "KB_EVAL_ENTRYPOINT"

DEFAULT_KB_PYTHON = "/raid/user_data/olu/venv/bin/python"
DEFAULT_KB_REPO = "/raid/user_data/olu/kb_agent_eval"
DEFAULT_KB_ENTRYPOINT = (
    "/raid/user_data/olu/scratch/agent_eval_pilot/entrypoint/agent_entrypoint.py"
)

# Definition names are `kb_<op>`; the entrypoint takes `--op <op>`.
DEFINITION_PREFIX = "kb_"

# Coupling to the entrypoint's error text: it raises InfraError("candidate import
# failed: ...") when the agent's kernel.py won't import, and exits 2. That is the
# agent's fault and must reach the agent as per-workload data (COMPILE_ERROR), not
# as an adapter crash. If the entrypoint's wording changes, update this marker —
# the fallback is a raised infrastructure error, which fails loud, not silent.
_CANDIDATE_IMPORT_MARKER = "candidate import failed"

_MAX_LOG_CHARS = 3000


# ===========================================================================
# Plain-data public surface (the data-contract seam)
# ===========================================================================

def list_workloads(dataset_path, definition):
    """Return ``[{"uuid": str, "axes": dict}, ...]`` for ``definition``, in dataset order."""
    entries = []
    for record in _read_workload_records(dataset_path, definition):
        wl = record["workload"]
        entries.append({"uuid": wl["uuid"], "axes": dict(wl.get("axes", {}))})
    return entries


def pack(source_dir, build_cfg, *, name, definition, author):
    """Pack kernel sources from ``source_dir`` into a solution-blob (JSON text).

    ``build_cfg`` keys: ``language`` (advisory metadata — kb always executes
    Python, whatever the kernel JITs underneath), ``entry_point``
    (``<file>::<func>``; only the file part is used — kb dispatches on the module's
    nn.Module class, not a function, so ``::run`` is vestigial),
    ``destination_passing_style`` (no kb analogue, dropped).

    bench_utils hardcodes ``{"language": "triton", "entry_point": "kernel.py::run",
    "destination_passing_style": False}`` when it packs the definition's reference;
    that must keep working, hence the tolerant treatment above.
    """
    src = Path(source_dir)
    if not src.is_dir():
        raise ValueError(f"pack(): source_dir is not a directory: {source_dir}")

    entry_point = build_cfg.get("entry_point") or "kernel.py::run"
    entry_file = entry_point.split("::", 1)[0].strip()

    sources = {}
    for path in sorted(src.rglob("*.py")):
        if any(part in ("__pycache__", ".ipynb_checkpoints") for part in path.parts):
            continue
        sources[path.relative_to(src).as_posix()] = path.read_text()

    if not sources:
        raise ValueError(f"pack(): no .py sources found under {source_dir}")
    if entry_file not in sources:
        raise ValueError(
            f"pack(): entry file {entry_file!r} (from entry_point {entry_point!r}) "
            f"not among the packed sources {sorted(sources)}"
        )

    return json.dumps({
        "name": name,
        "definition": definition,
        "author": author,
        "entry_file": entry_file,
        "language": build_cfg.get("language", "python"),
        "sources": sources,
    }, indent=2)


def solution_meta(blob):
    """``{"name", "definition", "author"}`` from a solution-blob.

    The single sanctioned place that introspects a blob's internals, so callers
    can treat the blob as opaque.
    """
    sol = json.loads(blob)
    missing = [k for k in ("name", "definition", "author") if k not in sol]
    if missing:
        raise ValueError(f"solution blob is missing required key(s): {missing}")
    return {"name": sol["name"], "definition": sol["definition"], "author": sol["author"]}


def run(blob, uuids, params, *, dataset_path, capture_logs=False, capture_autotune=False):
    """Run the kb entrypoint over the workloads named by ``uuids``.

    Materializes the blob's sources in a tempdir, invokes

        $KB_EVAL_PYTHON $KB_EVAL_ENTRYPOINT --op <op> --candidate <tmp>/<entry_file>
                        --scenarios <uuid,uuid,...>

    with ``PYTHONPATH=<tmpdir>:$KB_EVAL_REPO`` (and the caller's environment,
    including ``CUDA_VISIBLE_DEVICES``, passed through), parses the one JSON object
    on stdout, and returns the normalized result dict. ``capture_autotune=True``
    returns ``{"results": <dict>, "autotune_log": <str>}``; kb emits no autotune
    log, so that string is always empty.
    """
    sol = json.loads(blob)
    definition = sol["definition"]
    op = _op_from_definition(definition)

    if not uuids:
        raise ValueError("run() called with no workload uuids — nothing to benchmark")
    requested = list(dict.fromkeys(uuids))

    # Enforce exactly-the-requested uuids against the dataset BEFORE running. The
    # caller resolves uuids from docs/workloads.jsonl; this checks them against the
    # dataset at dataset_path. If those two diverge (stale docs, wrong-definition
    # uuid), a membership filter would silently run a SUBSET and corrupt the
    # score/baseline (compute_score averages over whatever ran). Raise instead.
    known = {w["uuid"]: w["axes"] for w in list_workloads(dataset_path, definition)}
    missing = [u for u in requested if u not in known]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(requested)} requested workload uuid(s) not found in "
            f"the dataset for definition '{definition}' (e.g. {sorted(missing)[0]!r}). "
            f"The selection source (docs/workloads.jsonl) and the execution dataset "
            f"({dataset_path}) may have diverged."
        )

    timeout = _subprocess_timeout(params, len(requested))
    solution_name = sol.get("name", "solution")

    with tempfile.TemporaryDirectory(prefix="ako_kb_solution_") as tmp_dir:
        candidate = _materialize(sol, tmp_dir)
        cmd = [
            _kb_python(), _kb_entrypoint(),
            "--op", op,
            "--candidate", candidate,
            "--scenarios", ",".join(requested),
        ]
        env = _child_env(tmp_dir)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                  env=env)
        except subprocess.TimeoutExpired:
            results = {definition: {
                u: {"status": STATUS_TIMEOUT, "solution": solution_name,
                    "axes": dict(known[u]),
                    "error_log": (f"kb entrypoint exceeded {timeout:g}s for "
                                  f"{len(requested)} workload(s) "
                                  f"(timeout_seconds x n_workloads)")}
                for u in requested}}
            return {"results": results, "autotune_log": ""} if capture_autotune else results

    stderr = proc.stderr or ""
    if proc.returncode != 0:
        if _CANDIDATE_IMPORT_MARKER in stderr:
            log = _truncate_log(stderr)
            results = {definition: {
                u: {"status": STATUS_COMPILE_ERROR, "solution": solution_name,
                    "axes": dict(known[u]), "error_log": log}
                for u in requested}}
            return {"results": results, "autotune_log": ""} if capture_autotune else results
        raise RuntimeError(
            f"kb entrypoint failed (exit {proc.returncode}) for definition "
            f"'{definition}'. Command: {' '.join(cmd)}\n--- stderr tail ---\n"
            f"{_truncate_log(stderr)}"
        )

    results = _parse_results(proc.stdout, definition, requested, known,
                             solution_name, stderr, capture_logs)
    return {"results": results, "autotune_log": ""} if capture_autotune else results


# --- Optional tier (stubs; only the named command degrades) -------------------

def profile(blob, uuid, opts, *, dataset_path, env_pairs=None):
    return "NCU profiling not supported for kb-nano (adapter stub)"


def list_ncu_options():
    return "NCU not supported for kb-nano (adapter stub)"


def sanitize(blob, uuid, opts, *, dataset_path):
    return "compute-sanitizer not supported for kb-nano (adapter stub)"


def cheat_check(blob, uuids, *, dataset_path, n_iters=4):
    return {"status": "SKIPPED",
            "reason": "cheat-check not supported for kb-nano (adapter stub)"}


# --- adapter-private helpers (not part of the public surface) ----------------

def _kb_python():
    return os.environ.get(KB_PYTHON_ENV) or DEFAULT_KB_PYTHON


def _kb_repo():
    return os.environ.get(KB_REPO_ENV) or DEFAULT_KB_REPO


def _kb_entrypoint():
    return os.environ.get(KB_ENTRYPOINT_ENV) or DEFAULT_KB_ENTRYPOINT


def _op_from_definition(definition):
    """``kb_rms_norm`` -> ``rms_norm`` (the entrypoint's ``--op``)."""
    if not definition.startswith(DEFINITION_PREFIX):
        raise ValueError(
            f"definition {definition!r} does not start with {DEFINITION_PREFIX!r}; "
            f"this adapter only serves kb operators (definition name = 'kb_<op>')"
        )
    op = definition[len(DEFINITION_PREFIX):]
    if not op:
        raise ValueError(f"definition {definition!r} has an empty operator name")
    return op


def _read_workload_records(dataset_path, definition):
    """Read the dataset's workloads.jsonl for ``definition`` (Trace-envelope lines)."""
    op = _op_from_definition(definition)
    root = Path(dataset_path)
    matches = sorted(root.glob(f"workloads/*/{definition}.jsonl"))
    if not matches:
        raise ValueError(
            f"no workloads file for definition '{definition}' under {root}/workloads/"
            f"<op_type>/{definition}.jsonl (operator '{op}')"
        )
    records = []
    with open(matches[0]) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _subprocess_timeout(params, n_workloads):
    """Per-workload ``timeout_seconds`` scaled to the whole-subprocess budget."""
    raw = (params or {}).get("timeout_seconds")
    try:
        per_workload = float(raw)
    except (TypeError, ValueError):
        return None
    if per_workload <= 0:
        return None
    return per_workload * max(1, n_workloads)


def _materialize(sol, tmp_dir):
    """Write the blob's sources into ``tmp_dir``; return the absolute entry path."""
    sources = sol.get("sources") or {}
    entry_file = sol.get("entry_file") or "kernel.py"
    if entry_file not in sources:
        raise ValueError(
            f"solution blob's entry_file {entry_file!r} is not among its sources "
            f"{sorted(sources)}"
        )
    root = Path(tmp_dir)
    for rel, content in sources.items():
        target = root / rel
        # Refuse path escapes from a hand-edited blob.
        if not str(target.resolve()).startswith(str(root.resolve()) + os.sep):
            raise ValueError(f"solution blob source path escapes the staging dir: {rel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return str(root / entry_file)


def _child_env(tmp_dir):
    """Caller env + kb PYTHONPATH. CUDA_VISIBLE_DEVICES rides along untouched."""
    env = os.environ.copy()
    # tmp_dir first so a solution's helper modules import; the kb repo second so
    # the entrypoint's tree bootstrap finds it (it scans PYTHONPATH entries).
    env["PYTHONPATH"] = os.pathsep.join([tmp_dir, _kb_repo()])
    env["FASTKERNELS_TREE"] = _kb_repo()
    return env


def _truncate_log(log, max_chars=_MAX_LOG_CHARS):
    """Truncate log to the last max_chars characters, preserving line boundaries."""
    if not log or len(log) <= max_chars:
        return log
    truncated = log[-max_chars:]
    nl = truncated.find("\n")
    if nl != -1 and nl < 200:
        truncated = truncated[nl + 1:]
    return f"[...truncated...]\n{truncated}"


def _parse_results(stdout, definition, requested, known, solution_name, stderr,
                   capture_logs):
    """Validate the entrypoint's JSON and normalize it into the contract shape."""
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError(
            f"kb entrypoint exited 0 but printed nothing on stdout for definition "
            f"'{definition}'.\n--- stderr tail ---\n{_truncate_log(stderr)}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"kb entrypoint stdout is not one JSON object ({exc}) for definition "
            f"'{definition}'.\n--- stdout head ---\n{text[:1000]}"
            f"\n--- stderr tail ---\n{_truncate_log(stderr)}"
        ) from exc

    if list(payload) != [definition]:
        raise RuntimeError(
            f"kb entrypoint returned definition key(s) {list(payload)}, expected "
            f"exactly ['{definition}']"
        )
    traces = payload[definition]

    # The entrypoint must have run exactly what we asked for. A subset would
    # silently shrink the score's denominator; a superset means the scenario
    # filter matched by substring instead of exactly.
    returned = set(traces)
    if returned != set(requested):
        raise RuntimeError(
            f"kb entrypoint ran a different workload set than requested for "
            f"'{definition}': missing={sorted(set(requested) - returned)}, "
            f"unexpected={sorted(returned - set(requested))}"
        )

    log = _truncate_log(stderr, max_chars=20000) if capture_logs else None
    normalized = {}
    for uuid in requested:                      # preserve requested order
        entry = dict(traces[uuid])
        status = entry.get("status")
        if status not in _VALID_STATUSES:
            raise RuntimeError(
                f"kb entrypoint returned unknown status {status!r} for workload "
                f"{uuid!r}; expected one of {sorted(_VALID_STATUSES)}"
            )
        # The entrypoint reports the candidate's staged path here (a tempdir), which
        # is meaningless downstream — restore the solution's identity instead.
        entry["solution"] = solution_name
        if not entry.get("axes"):
            entry["axes"] = dict(known[uuid])
        if status != STATUS_PASSED and not entry.get("error_log"):
            entry["error_log"] = _truncate_log(stderr)
        if log:
            entry["log"] = log
        normalized[uuid] = entry
    return {definition: normalized}
