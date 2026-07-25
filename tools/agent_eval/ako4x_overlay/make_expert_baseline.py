#!/usr/bin/env python3
"""Build `expert_baseline.json` (a pack()-format blob) for a kb child env.

`expert_baseline.json` is NOT part of the dataset — spawn.py never writes it.
It is copied into the CHILD ROOT after spawn; bench_utils.load_expert_blob()
finds it there and profiles it as the score denominator (the "expert" the agent
must beat) instead of the definition's own reference.

    python make_expert_baseline.py --out <child>/expert_baseline.json

The blob is produced by the adapter's own pack(), so its format cannot drift from
what run() expects.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark_adapter as adapter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "expert_baseline.json")
    ap.add_argument("--definition", default="kb_rms_norm")
    ap.add_argument("--source", type=Path, default=HERE / "expert_kernel.py")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "kernel.py").write_text(args.source.read_text())
        blob = adapter.pack(
            tmp,
            {"language": "python", "entry_point": "kernel.py::run",
             "destination_passing_style": False},
            name=f"{args.definition}-expert-baseline",
            definition=args.definition,
            author="baseline",
        )
    args.out.write_text(blob)
    meta = adapter.solution_meta(blob)
    print(f"wrote {args.out}")
    print(f"  name       : {meta['name']}")
    print(f"  definition : {meta['definition']}")
    print(f"  author     : {meta['author']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
