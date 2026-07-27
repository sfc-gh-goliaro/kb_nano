#!/usr/bin/env python3
"""Is a gpt-oss run's output well-formed, and which side diverges?

The token-agreement metric cannot distinguish "we changed" from "the reference changed".
For gpt-oss-120b on B200 the reference is the side that moved: it emits a leading space
instead of the harmony ``<|channel|>`` control token on ~94% of requests, where our output
is well-formed. That makes the raw match count useless for judging our correctness, so this
scores both sides against the harmony format itself rather than against each other.

gpt-oss emits OpenAI harmony: a completion should open with ``<|channel|>`` followed by a
channel name (``analysis`` or ``final``) and ``<|message|>``. Anything else -- a bare
space, a truncated ``fanalysis`` -- is malformed regardless of what the other side did.

    python harmony_check.py                          # newest run of each gpt-oss size
    python harmony_check.py <results_dir> [...]      # specific run directories
"""
from __future__ import annotations

import collections
import glob
import json
import pathlib
import sys

RESULTS = pathlib.Path("/home/yak/kb_nano/tests/results/B200")


def load(run: pathlib.Path, scenario: str, side: str):
    p = run / scenario / f"{side}_outputs.json"
    if not p.exists():
        return None
    try:
        return [o.get("token_ids") or [] for o in json.load(open(p))["outputs"]]
    except Exception:
        return None


def report(run: pathlib.Path, tok) -> None:
    scenarios = sorted(d.name for d in run.iterdir() if d.is_dir())
    print(f"\n=== {run.parent.name}/{run.name} ===")
    for sc in scenarios:
        ours, ref = load(run, sc, "fastkernels"), load(run, sc, "vllm")
        if not ours or not ref:
            continue
        n = min(len(ours), len(ref))
        stats = {}
        for label, seqs in (("ours", ours), ("reference", ref)):
            first = collections.Counter(
                tok.decode([s[0]]) if s else "<empty>" for s in seqs[:n])
            # Well-formed = opens the harmony envelope.
            wf = sum(1 for s in seqs[:n] if s and tok.decode([s[0]]) == "<|channel|>")
            stats[label] = (first.most_common(3), wf)
        div = []
        for i in range(n):
            a, b = ours[i], ref[i]
            div.append(next((k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                            min(len(a), len(b))))
        print(f"  {sc}  ({n} requests, mean divergence index {sum(div)/n:.1f})")
        for label, (common, wf) in stats.items():
            shown = ", ".join(f"{t!r}x{c}" for t, c in common)
            print(f"    {label:<9} well-formed {wf}/{n} ({wf/n:.0%})   first token: {shown}")


def main() -> int:
    from transformers import AutoTokenizer
    args = sys.argv[1:]
    runs = [pathlib.Path(a) for a in args]
    if not runs:
        for model in ("gpt-oss-20b_tp1", "gpt-oss-120b_tp1", "gpt-oss-120b_tp2"):
            cands = sorted(glob.glob(str(RESULTS / model / "*" / "results.json")),
                           key=lambda p: pathlib.Path(p).stat().st_mtime)
            if cands:
                runs.append(pathlib.Path(cands[-1]).parent)
    if not runs:
        print("no gpt-oss runs found")
        return 1
    # Both sizes share the harmony tokenizer; load once from whichever is present.
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    for r in runs:
        report(r, tok)
    print("\nA side scoring low here is producing malformed harmony, independently of\n"
          "what the other side did -- which is what the match count cannot tell you.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
