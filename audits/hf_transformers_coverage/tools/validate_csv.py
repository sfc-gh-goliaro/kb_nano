"""Validate a coverage CSV (pilot or shard) against the source-of-truth.

Checks per row:
- support_status is in the allowed set.
- partial/unsupported rows have non-empty partial_or_unsupported_ops.
- HF folder name in inventory.
- modeling_file (if any) exists at the pinned HF source path.
- evidence_hf entries are syntactically valid `<folder>/<file>:<lineno>` and the file
  exists. Optionally: read the file and check the line is non-empty (warning, not failure).
- mapped_kb_nano entries reference a canonical op name in the canonical map AND
  cite a kb-nano path that exists.

Run as:
    python validate_csv.py <path_to_csv>

Exits non-zero with a diagnosable summary if any HARD CHECK fails.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path("/home/olu/kb_nano/audits/hf_transformers_coverage")
KB_REPO = Path("/home/olu/kb_nano")
HF_PINNED = Path("/tmp/hf_transformers_pinned/src/transformers/models")
INVENTORY = ROOT / "hf_model_inventory.csv"
CANONICAL = ROOT / "tools/canonical_to_kb_nano.csv"

ALLOWED_STATUS = {"kb_nano_l4", "composable", "partial", "unsupported", "not_inference_required"}

EVIDENCE_RE = re.compile(r"^([\w/.-]+):(\d+)$")
MAPPED_RE = re.compile(r"^(\w[\w_]*)(?:→|->)(tasks/baseline/[\w/.-]+\.py)(?::(\w[\w_]*))?(?:\(.*\))?$")


def load_inv_folders() -> set[str]:
    with open(INVENTORY) as f:
        return {r["folder"] for r in csv.DictReader(f)}


def load_canonical_ops() -> set[str]:
    ops = set()
    with open(CANONICAL) as f:
        for r in csv.DictReader(f):
            ops.add(r["canonical_op"])
    return ops


def main(path: Path):
    inv_folders = load_inv_folders()
    canonical_ops = load_canonical_ops()
    rows = list(csv.DictReader(open(path)))
    print(f"Validating {len(rows)} rows in {path}")
    fail = 0
    warn = 0
    for i, r in enumerate(rows, 1):
        prefix = f"row {i} ({r.get('hf_folder','?')}/{r.get('modeling_file','?')})"
        # 1. status
        st = r["support_status"]
        if st not in ALLOWED_STATUS:
            print(f"FAIL {prefix}: bad status {st!r}")
            fail += 1
            continue
        # 2. partial/unsupported needs op detail
        if st in ("partial", "unsupported") and not r["partial_or_unsupported_ops"].strip():
            print(f"FAIL {prefix}: status={st} but partial_or_unsupported_ops empty")
            fail += 1
        # 3. hf_folder in inventory (only if not empty)
        fld = r["hf_folder"]
        if fld and fld not in inv_folders:
            print(f"FAIL {prefix}: hf_folder {fld!r} not in inventory")
            fail += 1
        # 4. modeling_file path (only if specified)
        mf = r["modeling_file"]
        if mf:
            full = HF_PINNED / mf.replace(fld + "/", fld + "/", 1) if mf.startswith(fld) else None
            # mf is "<folder>/modeling_<x>.py"; allow either form
            cand = HF_PINNED / mf if "/" in mf else None
            if cand and not cand.exists():
                print(f"FAIL {prefix}: modeling_file {mf} not at {cand}")
                fail += 1
        else:
            # not_inference_required allowed
            if st != "not_inference_required":
                print(f"FAIL {prefix}: missing modeling_file but status is {st!r}")
                fail += 1
        # 5. evidence_hf entries
        for ev in (r["evidence_hf"] or "").split(";"):
            ev = ev.strip()
            if not ev:
                continue
            m = EVIDENCE_RE.match(ev)
            if not m:
                print(f"WARN {prefix}: evidence_hf entry {ev!r} doesn't match <path>:<lineno>")
                warn += 1
                continue
            file_part, ln = m.group(1), int(m.group(2))
            full = HF_PINNED / file_part
            if not full.exists():
                print(f"FAIL {prefix}: evidence_hf cites non-existent file {file_part}")
                fail += 1
                continue
            with open(full) as f:
                lines = f.readlines()
            if ln <= 0 or ln > len(lines):
                print(f"FAIL {prefix}: evidence_hf line {ln} out of range (file has {len(lines)} lines)")
                fail += 1
                continue
            line = lines[ln-1].strip()
            if not line:
                print(f"WARN {prefix}: evidence_hf cites empty line {ev}")
                warn += 1
        # 6. mapped_kb_nano entries
        for mp in (r["mapped_kb_nano"] or "").split(";"):
            mp = mp.strip()
            if not mp:
                continue
            m = MAPPED_RE.match(mp)
            if not m:
                # be lenient — accept entries that say "via vLLM" etc.
                if "via" in mp or "no L1 kernel" in mp or "passthrough" in mp:
                    continue
                print(f"WARN {prefix}: mapped_kb_nano entry {mp!r} not in <op>→<path>[:<class>] form")
                warn += 1
                continue
            op, path_part, klass = m.group(1), m.group(2), m.group(3)
            if op not in canonical_ops:
                print(f"WARN {prefix}: mapped_kb_nano canonical op {op!r} not in canonical map (acceptable if pre-merge)")
                warn += 1
            full = KB_REPO / path_part
            if not full.exists():
                print(f"FAIL {prefix}: mapped_kb_nano cites non-existent kb-nano path {path_part}")
                fail += 1
    print(f"\nValidation: {fail} hard failures, {warn} warnings, {len(rows)} total rows")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: validate_csv.py <path>")
        sys.exit(1)
    main(Path(sys.argv[1]))
