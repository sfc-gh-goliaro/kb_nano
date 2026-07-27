#!/usr/bin/env python3
"""Free specific GPUs by index, and only those.

Manual cleanup of wedged jobs kept reaching for `pkill -f <name>`, but benchmark
workers appear in `ps` as `python -u /tmp/tmpXXXX.py /tmp/YYYY.json` -- the model
name is nowhere in the cmdline. A name-based guard therefore cannot protect a job
you want to keep, and on one occasion this nearly killed a HunyuanVideo run that
had been going for 80 minutes (it happened to finish a minute earlier).

Select by GPU index instead: map index -> UUID via nvidia-smi, then kill only the
compute processes resident on those UUIDs.

    python free_gpus.py 1 2 4 5        # free these, leave everything else alone
    python free_gpus.py --list        # show what is on each GPU
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def smi(query: str, extra: list[str] | None = None) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{query.split(':')[0]}={query.split(':')[1]}",
         "--format=csv,noheader", *(extra or [])],
        capture_output=True, text=True)
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def uuid_by_index() -> dict[int, str]:
    m = {}
    for ln in smi("gpu:index,uuid"):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            m[int(parts[0])] = parts[1]
    return m


def procs_on(uuids: set[str]) -> list[tuple[int, str, str]]:
    rows = []
    for ln in smi("compute-apps:pid,gpu_uuid,used_memory"):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and parts[1] in uuids:
            rows.append((int(parts[0]), parts[2], parts[1]))
    return rows


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")[:110]
    except OSError:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpus", nargs="*", type=int)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    idx = uuid_by_index()
    if a.list or not a.gpus:
        for i, u in sorted(idx.items()):
            on = procs_on({u})
            desc = ", ".join(f"pid {p} ({m}) {cmdline(p)}" for p, m, _ in on) or "idle"
            print(f"GPU {i}: {desc}")
        return 0

    targets = {idx[g] for g in a.gpus if g in idx}
    victims = procs_on(targets)
    if not victims:
        print(f"GPUs {a.gpus} already free")
        return 0
    for pid, mem, _ in victims:
        print(f"  killing pid {pid} ({mem}) {cmdline(pid)}")
    for pid, _, _ in victims:
        # Kill the whole process group so TP workers go with their parent.
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for _ in range(8):
        time.sleep(3)
        if not procs_on(targets):
            break
    left = procs_on(targets)
    print("remaining on those GPUs:", left or "none")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
