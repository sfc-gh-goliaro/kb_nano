#!/usr/bin/env python3
"""Compare B200 token-level alignment against the paper's Align./Value column.

compare.py answers "did the speedup reproduce". This answers the other half of the
table, which matters just as much for the rebuttal: the paper quotes an absolute
"avg toks" agreement per row (e.g. Llama-3.1 408.5 tok), and a row can hit its
speedup while agreeing with the reference far less than it did on H200.

Rules, learned the hard way from compare.py:
  * pick one run per model -- the newest one at the largest num_seqs seen, so a
    16-seq diagnostic never masquerades as the headline number;
  * report absolute matched tokens (the paper's unit) *and* the fraction, since
    output lengths differ between runs.
"""
from __future__ import annotations

import json
import pathlib
import sys

RESULTS = pathlib.Path("/home/yak/kb_nano/tests/results/B200")

# Paper's Align./Value column, avg-toks rows only (Top-20 / cos rows are scored by
# a different script and are not comparable here).
PAPER = {
    "Llama-3.1-8B-Instruct": 408.5,
    "DeepSeek-V3.2": 294.1,
    "Mixtral-8x7B-Instruct-v0.1": 108.9,
    "gpt-oss-20b": 599.6,
    "gpt-oss-120b": 599.6,
    "Mamba-Codestral-7B-v0.1": 541.3,
    "mamba-2.8b-hf": 555.9,
    "rwkv7-2.9B-g1": 593.8,
    "gla-2.7B-100B": 645.5,
    "retnet-2.7B-100B": 647.0,
    "Qwen3-Next-80B-A3B-Instruct": 487.4,
    "AI21-Jamba-Mini-1.7": 415.4,
    "Qwen2-VL-7B-Instruct": 539.4,
    "Qwen3-VL-8B-Instruct": 368.5,
    "whisper-large-v3": 388.7,
}


def runs():
    """model -> best run (largest num_seqs, then newest) -> per-scenario alignment."""
    best: dict[str, tuple] = {}
    for p in RESULTS.rglob("results.json"):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if not isinstance(r, dict) or not isinstance(r.get("scenarios"), list):
            continue
        rows = []
        for sc in r["scenarios"]:
            al = sc.get("alignment") or {}
            a, o = (al.get("avg_matching_tokens_per_request"),
                    al.get("avg_output_len"))
            if a is not None and o:
                rows.append((sc.get("scenario", "?"), float(a), float(o)))
        if not rows:
            continue
        model = (r.get("model") or "?").split("/")[-1]
        # num_seqs is absent on older runs; fall back to the reported seq count.
        n = r.get("num_seqs") or max(
            (sc.get("num_seqs") or 0) for sc in r["scenarios"])
        key = (n or 0, p.stat().st_mtime)
        if model not in best or key > best[model][0]:
            best[model] = (key, p, rows)
    return best


def main() -> int:
    best = runs()
    print(f"{'model':<32} {'n':>5}  {'B200 avg toks (per scenario)':<30} "
          f"{'mean':>7} {'paper':>7} {'ratio':>6}")
    print("-" * 96)
    behind = []
    for model, ((n, _), path, rows) in sorted(best.items()):
        per = " ".join(f"{a:.0f}/{o:.0f}" for _, a, o in rows)
        mean = sum(a for _, a, _ in rows) / len(rows)
        tgt = PAPER.get(model)
        ratio = f"{mean / tgt:.2f}" if tgt else "-"
        print(f"{model[:31]:<32} {n:>5}  {per:<30} {mean:>7.1f} "
              f"{tgt if tgt else '-':>7} {ratio:>6}")
        if tgt and mean < 0.5 * tgt:
            behind.append((model, mean, tgt))
    if behind:
        print("\nrows agreeing with the reference far less than the paper:")
        for m, mean, tgt in sorted(behind, key=lambda x: x[1] / x[2]):
            print(f"  {m:<32} {mean:7.1f} vs {tgt:7.1f}  ({mean / tgt:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
