#!/usr/bin/env python3
"""Simple GPU-aware job scheduler for the B200 reproduction sweep.

Reads a JSON job list: [{"name": ..., "cmd": ..., "gpus": 1, "timeout": 7200}, ...]
Allocates contiguous GPU ids from a pool, runs each job with CUDA_VISIBLE_DEVICES
set, and writes a per-job log plus a summary JSON.

Usage:
    python sched.py jobs.json [--pool 0,1,2,3,4,5,6,7] [--only name1,name2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOGDIR = Path("/home/yak/b200_repro/logs/bench")
LOGDIR.mkdir(parents=True, exist_ok=True)
STATUS = Path("/home/yak/b200_repro/logs/status.json")

# A GPU with more than this in use is treated as busy. Benchmarks size their KV
# cache from *free* memory, so launching onto a GPU still holding a dead job's
# allocation silently changes the measurement instead of failing loudly.
FREE_MIB = 2048
DRAIN_TIMEOUT = 90
# A job whose log has not grown for this long while its GPUs sit idle is wedged.
# Benchmarks that hit a fatal CUDA error under tensor parallelism routinely print
# the traceback and then hang forever in NCCL/process-group teardown, holding
# whole GPUs until the job timeout (hours) expires.
STALL_SECS = 600
STALL_UTIL_PCT = 5


def gpu_util_pct(ids):
    rows = _nvidia_smi(("gpu", "index,utilization.gpu"))
    util = {}
    for ln in rows:
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                util[int(parts[0])] = int(float(parts[1]))
            except ValueError:
                pass
    return {i: util.get(i, 0) for i in ids}


def _nvidia_smi(query, extra=()):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query[0]}={query[1]}",
             "--format=csv,noheader,nounits", *extra],
            capture_output=True, text=True, timeout=30)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def gpu_used_mib(ids):
    """Used MiB per GPU id, as a dict. Missing ids are reported as 0."""
    rows = _nvidia_smi(("gpu", "index,memory.used"))
    used = {}
    for ln in rows:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            used[int(parts[0])] = int(float(parts[1]))
    return {i: used.get(i, 0) for i in ids}


def gpus_free(ids):
    return all(v <= FREE_MIB for v in gpu_used_mib(ids).values())


def kill_stragglers(ids, log_fn):
    """Kill compute processes still resident on ``ids`` after a job exited.

    The scheduler owns GPU assignment exclusively, so anything left on a
    finished job's GPUs is that job's orphan (e.g. TP workers whose parent died
    without reaping them).
    """
    uuid_of = {}
    for ln in _nvidia_smi(("gpu", "index,uuid")):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            uuid_of[int(parts[0])] = parts[1]
    targets = {uuid_of.get(i) for i in ids} - {None}
    victims = []
    for ln in _nvidia_smi(("compute-apps", "pid,gpu_uuid")):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1] in targets:
            victims.append(int(parts[0]))
    for pid in victims:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    if victims:
        log_fn(f"[sched] killed {len(victims)} straggler pid(s) on GPUs {ids}: {victims}")


def drain(ids, log_fn):
    """Wait for a finished job's GPUs to actually free up, then force it."""
    deadline = time.time() + DRAIN_TIMEOUT
    while time.time() < deadline:
        if gpus_free(ids):
            return
        time.sleep(5)
    kill_stragglers(ids, log_fn)
    for _ in range(6):
        if gpus_free(ids):
            return
        time.sleep(5)
    log_fn(f"[sched] WARNING GPUs {ids} still busy after drain: {gpu_used_mib(ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--pool", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip jobs already recorded rc=0 in --status")
    args = ap.parse_args()

    status_path = Path(args.status) if args.status else STATUS
    jobs = json.loads(Path(args.jobs).read_text())
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["name"] in keep]
    if args.skip:
        drop = set(args.skip.split(","))
        jobs = [j for j in jobs if j["name"] not in drop]

    # Carry successful jobs across restarts so a watchdog relaunch is cheap.
    done = []
    if args.resume and status_path.exists():
        try:
            prev = json.loads(status_path.read_text())
        except Exception:
            prev = {}
        ok = {d["name"] for d in prev.get("done", []) if d.get("rc") == 0}
        done = [d for d in prev.get("done", []) if d.get("rc") == 0]
        if args.only:
            # --only is an explicit instruction (next_jobs.py decides what needs
            # running, including rows that exited 0 but produced no usable
            # reference). Filtering those back out by rc==0 would silently undo
            # every forced re-run, which is what happened to EAGLE-3, TTT-E2E,
            # CosyVoice3, V-JEPA 2 and DLRMv2. Keep only the carried-over `done`
            # list so status is preserved across restarts.
            print(f"[sched] resume: honouring --only ({len(jobs)} job(s)); "
                  f"carrying {len(done)} completed record(s)", flush=True)
        else:
            before = len(jobs)
            jobs = [j for j in jobs if j["name"] not in ok]
            print(f"[sched] resume: skipping {before - len(jobs)} completed jobs",
                  flush=True)

    free = [int(x) for x in args.pool.split(",")]
    pending = list(jobs)
    running = []  # (job, popen, gpus, t0, logfile)

    def dump():
        status_path.write_text(json.dumps({
            "running": [{"name": j["name"], "gpus": g,
                         "elapsed": round(time.time() - t0)}
                        for (j, p, g, t0, lf) in running],
            "pending": [j["name"] for j in pending],
            "done": done,
        }, indent=2))

    print(f"[sched] {len(jobs)} jobs, GPU pool {free}", flush=True)
    # A job wanting more GPUs than the pool holds can never be placed, and the loop
    # below would spin on it forever with the whole pool idle. This has happened twice:
    # the TP=8 rows against a 7-GPU pool (GPU 0 reserved for BitNet), and Mixtral's TP=4
    # against a 3-wide measurement pool. Reject them up front instead.
    unschedulable = [j for j in pending if j.get("gpus", 1) > len(free)]
    for job in unschedulable:
        print(f"[sched] SKIP  {job['name']}: needs {job['gpus']} GPUs, pool has "
              f"{len(free)} -- unschedulable, not spinning on it", flush=True)
        done.append({"name": job["name"], "rc": -97, "secs": 0})
        pending.remove(job)
    while pending or running:
        # launch what fits
        launched = True
        while launched:
            launched = False
            for i, job in enumerate(pending):
                n = job.get("gpus", 1)
                if len(free) >= n:
                    gpus = free[:n]
                    if not gpus_free(gpus):
                        # Leaked memory from a previous job: reclaim, don't measure.
                        print(f"[sched] GPUs {gpus} not free "
                              f"({gpu_used_mib(gpus)}); draining before "
                              f"{job['name']}", flush=True)
                        drain(gpus, lambda m: print(m, flush=True))
                        if not gpus_free(gpus):
                            break
                    del free[:n]
                    logf = LOGDIR / f"{job['name']}.log"
                    env = dict(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
                    env.update(job.get("env", {}))
                    fh = open(logf, "w")
                    fh.write(f"### CMD: {job['cmd']}\n### GPUS: {gpus}\n\n")
                    fh.flush()
                    p = subprocess.Popen(
                        job["cmd"], shell=True, stdout=fh, stderr=subprocess.STDOUT,
                        cwd=job.get("cwd", "/home/yak/kb_nano"), env=env,
                        preexec_fn=os.setsid,
                    )
                    running.append((job, p, gpus, time.time(), fh))
                    pending.pop(i)
                    print(f"[sched] START {job['name']} on GPUs {gpus}", flush=True)
                    launched = True
                    break
        dump()
        time.sleep(10)
        # reap
        for entry in list(running):
            job, p, gpus, t0, fh = entry
            rc = p.poll()
            el = time.time() - t0
            if rc is None:
                # Stall check: no log output for STALL_SECS and idle GPUs.
                try:
                    quiet = time.time() - os.path.getmtime(fh.name)
                except OSError:
                    quiet = 0
                if quiet > STALL_SECS and all(
                        v <= STALL_UTIL_PCT for v in gpu_util_pct(gpus).values()):
                    print(f"[sched] STALLED {job['name']}: no output for "
                          f"{quiet:.0f}s with GPUs {gpus} idle -- killing",
                          flush=True)
                    try:
                        os.killpg(os.getpgid(p.pid), 9)
                    except Exception:
                        pass
                    rc = -98
                    p.wait()
            if rc is None and el > job.get("timeout", 10800):
                print(f"[sched] TIMEOUT {job['name']} after {el:.0f}s", flush=True)
                try:
                    os.killpg(os.getpgid(p.pid), 9)
                except Exception:
                    pass
                rc = -99
                p.wait()
            if rc is not None:
                fh.close()
                running.remove(entry)
                drain(gpus, lambda m: print(m, flush=True))
                free.extend(gpus)
                free.sort()
                done.append({"name": job["name"], "rc": rc, "secs": round(el)})
                print(f"[sched] DONE  {job['name']} rc={rc} in {el:.0f}s", flush=True)
                dump()
    dump()
    print("[sched] ALL DONE", flush=True)
    for d in done:
        print(f"  {d['name']:<40} rc={d['rc']:<5} {d['secs']}s")


if __name__ == "__main__":
    main()
