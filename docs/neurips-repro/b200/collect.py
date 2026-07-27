#!/usr/bin/env python3
"""Collect + summarize every results.json under tests/results/<gpu>/.

Schemas differ per bench script, so this probes a handful of common shapes:
per-scenario ``speedup``, ``*_tok_per_s`` pairs, ``images_per_second``, etc.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/yak/kb_nano/tests/results")


def _num(x):
    try:
        return float(x)
    except Exception:
        return None


def scenario_speedups(d):
    """Yield (label, speedup, extra) for whatever scenario list this file has."""
    out = []
    for key in ("scenarios", "throughput", "results", "workloads", "throughput_scenarios"):
        v = d.get(key)
        if isinstance(v, list) and v:
            for s in v:
                if not isinstance(s, dict):
                    continue
                name = (s.get("scenario") or s.get("name") or s.get("workload")
                        or s.get("label") or "?")
                sp = s.get("speedup") or s.get("ratio") or s.get("speedup_vs_reference")
                if sp is None:
                    # derive from throughput pairs
                    ours = ref = None
                    for k, val in s.items():
                        n = _num(val)
                        if n is None:
                            continue
                        lk = k.lower()
                        if any(t in lk for t in ("fastkernels", "kb_nano", "ours")) and \
                           any(t in lk for t in ("tok_per_s", "per_second", "throughput", "img", "utt", "tps")):
                            ours = n
                        if any(t in lk for t in ("vllm", "reference", "ref_", "baseline", "timm",
                                                 "diffusers", "sota", "fla_")) and \
                           any(t in lk for t in ("tok_per_s", "per_second", "throughput", "img", "utt", "tps")):
                            ref = n
                    if ours and ref:
                        sp = ours / ref
                align = s.get("alignment") or {}
                extra = ""
                if isinstance(align, dict):
                    for ak in ("avg_matching_tokens_per_request", "mean_cosine",
                               "mean_cos_sim", "min_cosine", "match_rate"):
                        if ak in align:
                            extra = f"{ak}={align[ak]}"
                            break
                out.append((name, _num(sp), extra))
            if out:
                return out
    return out


def main():
    gpu = sys.argv[1] if len(sys.argv) > 1 else "B200"
    base = ROOT / gpu
    files = sorted(glob.glob(str(base / "**" / "*.json"), recursive=True))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        model = d.get("model") or d.get("model_name") or Path(f).parent.name
        tag = os.path.relpath(f, base)
        sc = scenario_speedups(d)
        if not sc:
            continue
        rows.append((model, tag, sc, os.path.getmtime(f)))

    rows.sort(key=lambda r: r[0].lower())
    print(f"{'MODEL':<48} {'SCENARIO':<26} {'SPEEDUP':>8}  ALIGN")
    print("-" * 110)
    for model, tag, sc, _mt in rows:
        sps = [s for (_n, s, _e) in sc if s]
        mean = sum(sps) / len(sps) if sps else None
        for i, (name, sp, extra) in enumerate(sc):
            m = model if i == 0 else ""
            spf = f"{sp:.3f}x" if sp else "   -  "
            print(f"{m[:47]:<48} {str(name)[:25]:<26} {spf:>8}  {extra}")
        if mean and len(sps) > 1:
            print(f"{'':<48} {'MEAN':<26} {mean:.3f}x")
        print()


if __name__ == "__main__":
    main()
