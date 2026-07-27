#!/usr/bin/env python3
"""Pick the jobs a watchdog relaunch should run, and record the attempt.

Prints a comma-separated job list for ``sched.py --only``, or nothing when
everything has either succeeded or exhausted its retries. Bounding retries
matters for unattended runs: without it a permanently failing job makes the
watchdog relaunch the scheduler forever.

A job that exited 0 but produced no fastkernels-vs-reference speedup is treated
as unfinished. A bench can exit 0 with a failed reference side (e.g. the vLLM
baseline OOMing), leaving a result file that has our throughput and no baseline;
those rows must be re-run, not reported.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/yak/b200_repro")
RESULTS = Path("/home/yak/kb_nano/tests/results/B200")
MAX_ATTEMPTS = 3

# Defaults target the main sweep; pass a jobs file and status file to reuse this
# for the low-concurrency re-measurement pass.
JOBS_FILE = sys.argv[1] if len(sys.argv) > 1 else "jobs_b200_full.json"
STATUS_FILE = sys.argv[2] if len(sys.argv) > 2 else "logs/status_full.json"


def _speedups_fn():
    """Reuse compare.py's schema-tolerant speedup extraction."""
    spec = importlib.util.spec_from_file_location("_cmp", ROOT / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.speedups


def has_baseline(job) -> bool | None:
    """True/False if the job's result dir can be identified, else None."""
    m = re.search(r"--model\s+(\S+)", job.get("cmd", ""))
    if not m or not RESULTS.exists():
        return None
    key = m.group(1).rstrip("/").split("/")[-1].lower()
    dirs = [d for d in RESULTS.iterdir() if d.is_dir() and key in d.name.lower()]
    if not dirs:
        return None
    speedups = _speedups_fn()
    for d in dirs:
        # Not every bench writes "results.json": bench_dllm names its file after
        # the task and config, so matching only results.json declared LLaDA
        # baseline-less forever and re-ran it on every relaunch.
        for rj in d.rglob("*.json"):
            try:
                if speedups(json.load(open(rj))):
                    return True
            except Exception:
                continue
    return False


def load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def main() -> int:
    jobs = load(ROOT / JOBS_FILE, [])
    status = load(ROOT / STATUS_FILE, {})
    att_path = ROOT / "logs" / (Path(JOBS_FILE).stem + "_attempts.json")
    attempts = load(att_path, {})

    ok = {d["name"] for d in status.get("done", []) if d.get("rc") == 0}
    failed = {d["name"] for d in status.get("done", []) if d.get("rc") != 0}
    # Explicit rerun list, for rows whose result dir cannot be inferred from the
    # command (no --model argument) but which are known to have produced no
    # reference -- e.g. EAGLE-3, where SGLang died and the bench carried on.
    # Jobs already running outside the scheduler (kept alive across a relaunch
    # because they are long and expensive) must not be started a second time.
    skip_path = ROOT / "logs" / "skip_while_orphaned.txt"
    skip = set()
    if skip_path.exists():
        skip = {ln.strip() for ln in skip_path.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")}
    force_path = ROOT / "logs" / "force_rerun.txt"
    force = set()
    if force_path.exists():
        force = {ln.strip() for ln in force_path.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")}
    todo = []
    for j in jobs:
        n = j["name"]
        if n in skip:
            print(f"# {n}: running as an untracked orphan -> not relaunching",
                  file=sys.stderr)
            continue
        if n in ok and n not in force and has_baseline(j) is not False:
            continue
        if n in ok:
            why = "listed in force_rerun.txt" if n in force else "has no baseline"
            print(f"# {n}: exited 0 but {why} -> re-running", file=sys.stderr)
        # Only count attempts against jobs that actually ran and failed. A job
        # that has never run (scheduler restarted before reaching it) must not
        # burn retries, or a few unrelated restarts would exhaust the whole list.
        if n in failed and attempts.get(n, 0) >= MAX_ATTEMPTS:
            continue
        todo.append(n)

    if todo:
        for n in todo:
            if n in failed:
                attempts[n] = attempts.get(n, 0) + 1
        att_path.write_text(json.dumps(attempts, indent=2, sort_keys=True))
        print(",".join(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
