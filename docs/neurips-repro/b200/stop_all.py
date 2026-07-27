#!/usr/bin/env python3
"""Stop the B200 sweep cleanly and free every GPU.

Kept as a file rather than a shell one-liner because ``pgrep -f <pattern>`` also
matches the shell command that contains the pattern, which makes interactive
cleanup match itself and report phantom survivors.

    python stop_all.py [--restart]
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

SELF = {os.getpid(), os.getppid()}
PATTERNS = ("sched.py", "watchdog.sh", "tests/bench_", "VLLM::")


def procs():
    out = subprocess.run(["ps", "-eo", "pid,pgid,args"],
                         capture_output=True, text=True).stdout.splitlines()[1:]
    rows = []
    for ln in out:
        parts = ln.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, pgid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        rows.append((pid, pgid, parts[2]))
    return rows


def matching():
    hits = []
    for pid, pgid, args in procs():
        if pid in SELF or "stop_all.py" in args:
            continue
        if any(p in args for p in PATTERNS):
            hits.append((pid, pgid, args[:90]))
    return hits


def gpu_used():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used = {}
    for ln in out.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            used[int(parts[0])] = int(float(parts[1]))
    return used


def compute_pids():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true",
                    help="relaunch one scheduler + watchdog after cleanup")
    a = ap.parse_args()

    # Two passes: process groups first (kills whole job trees), then leftovers.
    for attempt in range(3):
        hits = matching()
        if not hits:
            break
        print(f"pass {attempt}: killing {len(hits)} process(es)")
        for pid, pgid, args in hits:
            print(f"  {pid:>8} pgid={pgid:<8} {args}")
        for _pid, pgid, _a in hits:
            if pgid not in SELF:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        for pid, _pgid, _a in hits:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(5)

    # Anything still holding GPU memory is an orphan of a job we just killed.
    for _ in range(6):
        used = gpu_used()
        busy = {g: m for g, m in used.items() if m > 2048}
        if not busy:
            break
        left = compute_pids()
        print(f"GPUs still busy {busy}; killing {len(left)} compute pid(s)")
        for pid in left:
            if pid not in SELF:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        time.sleep(5)

    print("final GPU used MiB:", gpu_used())
    print("survivors:", matching() or "none")

    if a.restart:
        root = "/home/yak/b200_repro"
        env = dict(os.environ)
        env["HF_HOME"] = "/home/yak/data-fast/huggingface"
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        log = open(f"{root}/logs/sched_full.log", "a")
        subprocess.Popen(
            [sys.executable, "sched.py", "jobs_b200_full.json",
             "--pool", "0,1,2,3,4,5,6,7",
             "--status", f"{root}/logs/status_full.json", "--resume"],
            cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True)
        time.sleep(3)
        subprocess.Popen(["bash", f"{root}/watchdog.sh"], cwd=root,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=env, start_new_session=True)
        print("relaunched scheduler + watchdog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
