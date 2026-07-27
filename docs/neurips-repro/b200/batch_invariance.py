#!/usr/bin/env python3
"""How batch-invariant is each row, ours vs the reference?

The alignment collapse on B200 turned out not to be a disagreement with the
reference so much as a disagreement with *ourselves*: on full-scale Llama-3.1 our
output for a given prompt starts to differ after ~30 tokens when the batch grows from
64 to 1000, where vLLM's is stable for ~500. This sweeps that measurement over every
row that happens to have runs at two different scales, to see how general it is.

For each model: take the smallest-scale and largest-scale run, and for the requests
they share, report the mean token index at which the same prompt's output begins to
differ between the two. High = batch-invariant.
"""
from __future__ import annotations

import json
import pathlib
import sys

RESULTS = pathlib.Path("/home/yak/kb_nano/tests/results/B200")


def divergence(a: list[list[int]], b: list[list[int]], n: int) -> float:
    tot = 0
    for i in range(n):
        x, y = a[i], b[i]
        tot += next((k for k in range(min(len(x), len(y))) if x[k] != y[k]),
                    min(len(x), len(y)))
    return tot / n


def load(run: pathlib.Path, scenario: str, side: str):
    p = run / scenario / f"{side}_outputs.json"
    if not p.exists():
        return None
    try:
        outs = json.load(open(p))["outputs"]
    except Exception:
        return None
    tok = [o.get("token_ids") or [] for o in outs]
    return tok if any(tok) else None


def main() -> int:
    # model -> list of (num_seqs, run_dir, scenarios)
    bymodel: dict[str, list] = {}
    for rj in RESULTS.rglob("results.json"):
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        if not isinstance(r, dict) or not isinstance(r.get("scenarios"), list):
            continue
        n = r.get("num_seqs") or 0
        scs = [sc.get("scenario") for sc in r["scenarios"] if sc.get("scenario")]
        if not n or not scs:
            continue
        bymodel.setdefault((r.get("model") or "?").split("/")[-1], []).append(
            (n, rj.parent, scs))

    print(f"{'model':<32} {'scales':>12}  {'ours':>8} {'reference':>10}  scenario")
    print("-" * 82)
    rows = []
    for model, runs in sorted(bymodel.items()):
        runs.sort(key=lambda x: x[0])
        lo, hi = runs[0], runs[-1]
        if lo[0] == hi[0]:
            continue                      # only one scale available
        shared = [s for s in lo[2] if s in hi[2]]
        for sc in shared:
            ours_lo, ours_hi = load(lo[1], sc, "fastkernels"), load(hi[1], sc, "fastkernels")
            ref_lo, ref_hi = load(lo[1], sc, "vllm"), load(hi[1], sc, "vllm")
            if not (ours_lo and ours_hi):
                continue
            n = min(len(ours_lo), len(ours_hi))
            o = divergence(ours_lo, ours_hi, n)
            r = (divergence(ref_lo, ref_hi, min(len(ref_lo), len(ref_hi)))
                 if ref_lo and ref_hi else None)
            print(f"{model[:31]:<32} {f'{lo[0]}->{hi[0]}':>12}  {o:>8.1f} "
                  f"{(f'{r:.1f}' if r is not None else '-'):>10}  {sc}")
            rows.append((model, o, r))
            break                          # one scenario per model is enough
    if rows:
        worse = [(m, o, r) for m, o, r in rows if r and o < 0.5 * r]
        print(f"\n{len(worse)}/{len(rows)} rows are markedly less batch-invariant "
              f"than their reference:")
        for m, o, r in sorted(worse, key=lambda x: x[1] / x[2]):
            print(f"  {m:<32} ours {o:7.1f} vs reference {r:7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
