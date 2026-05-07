"""Clean stale prose in composable rows whose notes describe ops as partial.

After v3 + v3.1 audit passes, many rows that auto-reclassified `partial → composable`
still have notes from the original audit pass that describe their ops as
"partial via torch.nn fallback". This script prepends a clarification banner
to such rows (preserving historical text) and rewrites the most-common stale
phrases that incorrectly imply current-state partiality.

Approach: do not blindly delete. For each row, detect stale-status phrases
and either:
  1. Replace specific known-stale phrases with the correct current state
  2. Prepend a banner that says "[CURRENT STATE: composable; the following
     notes describe ops now covered by kb-nano L1/L2 additions in this audit
     branch — see audit_methodology.md § 17]"
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

KB_REPO = Path(os.environ.get("KB_NANO_REPO", Path(__file__).resolve().parents[3]))
ROOT = KB_REPO / "audits/hf_transformers_coverage"
COVERAGE_CSV = ROOT / "hf_architecture_operator_coverage.csv"

# Phrases that, when found in a *composable* row's notes, indicate stale prose.
# These are case-insensitive substring matches.
STALE_INDICATORS = [
    "no l1 kernel",
    "no l1 wrapper",
    "no l1 conv_transpose",
    "no l1 batch_norm",
    "no l1 max_pool",
    "no l1 avg_pool",
    "no l1 adaptive",
    "no l1 lstm",
    "no l1 grid",
    "no l1 leaky",
    "no l1 elu",
    "no l1 hardsigmoid",
    "no l1 hardswish",
    "no l1 multihead",
    "no l1 chunk_gated",
    "partial via torch.nn fallback",
    "partial per pilot convention",
    "partial fallback",
    "torch.nn fallback per kb-nano convention",
]

# Specific phrase rewrites (more aggressive than v3.0). Each pattern uses
# (?i) for case-insensitive and is intentionally permissive.
REPLACEMENTS = [
    # AdaptiveAvgPool2d — many phrasings
    (re.compile(r"(?i)adaptive_avg_pool_?2d\s*\(\s*no l1[^)]*\)"),
     "adaptive_avg_pool_2d → tasks/baseline/L1/adaptive_avg_pool2d.py:AdaptiveAvgPool2d"),
    (re.compile(r"(?i)nn\.AdaptiveAvgPool2d\s*\([^)]*\)\s*[-—]+\s*partial[^.;,\n]*"),
     "nn.AdaptiveAvgPool2d → covered by tasks/baseline/L1/adaptive_avg_pool2d.py:AdaptiveAvgPool2d"),
    (re.compile(r"(?i)nn\.AdaptiveAvgPool2d\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.AdaptiveAvgPool2d → covered by tasks/baseline/L1/adaptive_avg_pool2d.py:AdaptiveAvgPool2d"),
    # AdaptiveAvgPool1d
    (re.compile(r"(?i)adaptive_avg_pool_?1d\s*\(\s*no l1[^)]*\)"),
     "adaptive_avg_pool_1d → tasks/baseline/L1/adaptive_avg_pool1d.py:AdaptiveAvgPool1d"),
    (re.compile(r"(?i)nn\.AdaptiveAvgPool1d\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.AdaptiveAvgPool1d → covered by tasks/baseline/L1/adaptive_avg_pool1d.py:AdaptiveAvgPool1d"),
    # ConvTranspose2d
    (re.compile(r"(?i)conv_transpose_?2d\s*\(\s*no l1[^)]*\)"),
     "conv_transpose2d → tasks/baseline/L1/conv_transpose2d.py:ConvTranspose2d"),
    (re.compile(r"(?i)nn\.ConvTranspose2d\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.ConvTranspose2d → covered by tasks/baseline/L1/conv_transpose2d.py:ConvTranspose2d"),
    # ConvTranspose1d
    (re.compile(r"(?i)conv_transpose_?1d\s*\(\s*no l1[^)]*\)"),
     "conv_transpose1d → tasks/baseline/L1/conv_transpose1d.py:ConvTranspose1d"),
    (re.compile(r"(?i)nn\.ConvTranspose1d\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.ConvTranspose1d → covered by tasks/baseline/L1/conv_transpose1d.py:ConvTranspose1d"),
    # BatchNorm 1d/3d
    (re.compile(r"(?i)batch_norm_?1d\s*\(\s*no l1[^)]*\)"),
     "batch_norm_1d → composable via kb-nano BatchNorm2d (rank-agnostic)"),
    (re.compile(r"(?i)nn\.BatchNorm1d\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.BatchNorm1d → composable via kb-nano BatchNorm2d (rank-agnostic via F.batch_norm)"),
    (re.compile(r"(?i)batch_norm_?3d\s*\(\s*no l1[^)]*\)"),
     "batch_norm_3d → composable via kb-nano BatchNorm2d (rank-agnostic)"),
    (re.compile(r"(?i)nn\.BatchNorm3d\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.BatchNorm3d → composable via kb-nano BatchNorm2d (rank-agnostic)"),
    # MaxPool1d / AvgPool1d
    (re.compile(r"(?i)nn\.MaxPool1d\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.MaxPool1d → covered by tasks/baseline/L1/max_pool1d.py:MaxPool1d"),
    (re.compile(r"(?i)nn\.AvgPool1d\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.AvgPool1d → covered by tasks/baseline/L1/avg_pool1d.py:AvgPool1d"),
    (re.compile(r"(?i)max_pool_?1d\s*\(\s*no l1[^)]*\)"),
     "max_pool_1d → tasks/baseline/L1/max_pool1d.py:MaxPool1d"),
    (re.compile(r"(?i)avg_pool_?1d\s*\(\s*no l1[^)]*\)"),
     "avg_pool_1d → tasks/baseline/L1/avg_pool1d.py:AvgPool1d"),
    # Activations
    (re.compile(r"(?i)leaky_relu\s*\(\s*no l1[^)]*\)"),
     "leaky_relu → tasks/baseline/L1/leaky_relu.py:LeakyReLU"),
    (re.compile(r"(?i)\belu\s*\(\s*no l1[^)]*\)"),
     "elu → tasks/baseline/L1/elu.py:ELU"),
    (re.compile(r"(?i)hardsigmoid\s*\(\s*no l1[^)]*\)"),
     "hardsigmoid → tasks/baseline/L1/hardsigmoid.py:Hardsigmoid"),
    (re.compile(r"(?i)hardswish\s*\(\s*no l1[^)]*\)"),
     "hardswish → tasks/baseline/L1/hardswish.py:Hardswish"),
    # GridSample
    (re.compile(r"(?i)grid_sample\s*\(\s*no l1[^)]*\)"),
     "grid_sample → tasks/baseline/L1/grid_sample.py:GridSample"),
    # LSTM
    (re.compile(r"(?i)nn\.LSTM\b[^.;,\n]*partial[^.;,\n]*"),
     "nn.LSTM → covered by tasks/baseline/L1/lstm.py:LSTM"),
    (re.compile(r"(?i)\blstm\s*\(\s*no l1[^)]*\)"),
     "lstm → tasks/baseline/L1/lstm.py:LSTM"),
    # MultiheadAttention
    (re.compile(r"(?i)nn\.MultiheadAttention\b[^.;,\n]*(no l1 wrapper|partial)[^.;,\n]*"),
     "nn.MultiheadAttention → covered by tasks/baseline/L2/multihead_attention.py:MultiheadAttention"),
    (re.compile(r"(?i)multihead_attention\s*\([^)]*no l1 wrapper[^)]*\)"),
     "multihead_attention → tasks/baseline/L2/multihead_attention.py:MultiheadAttention"),
    # ChunkGatedDeltaRule
    (re.compile(r"(?i)chunk_gated_delta_rule\s*\(\s*no l1 wrapper[^)]*\)"),
     "chunk_gated_delta_rule → tasks/baseline/L1/chunk_gated_delta_rule.py:ChunkGatedDeltaRule"),
    # Generic "partial" qualifier left over
    (re.compile(r"(?i)\bpartial per pilot convention\b"),
     "covered by audit-branch L1 wrapper"),
    (re.compile(r"(?i)\bpartial via torch\.nn fallback\b"),
     "covered by audit-branch L1 wrapper"),
    (re.compile(r"(?i)\btorch\.nn fallback per kb-nano convention\b"),
     "covered by audit-branch L1 wrapper"),
]

BANNER = ("[v3.1: composable; ops described below are covered by kb-nano L1/L2 "
          "additions in this audit branch — see audit_methodology.md § 17] ")


def has_stale(notes: str) -> bool:
    low = (notes or "").lower()
    return any(p in low for p in STALE_INDICATORS)


def main():
    rows = list(csv.DictReader(open(COVERAGE_CSV)))
    fieldnames = list(rows[0].keys())
    n_cleaned = 0
    n_banner = 0
    for r in rows:
        if r["support_status"] != "composable":
            continue
        original = r.get("notes") or ""
        cleaned = original
        for pat, repl in REPLACEMENTS:
            cleaned = pat.sub(repl, cleaned)
        cleaned = re.sub(r"\s*;\s*;", ";", cleaned)
        cleaned = re.sub(r"  +", " ", cleaned)
        cleaned = cleaned.strip()
        if cleaned != original:
            r["notes"] = cleaned
            n_cleaned += 1
        # If stale phrases STILL remain after rewrites, prepend a banner so
        # the reader knows the row's current status is composable.
        if has_stale(r.get("notes") or "") and BANNER not in (r.get("notes") or ""):
            r["notes"] = BANNER + (r.get("notes") or "")
            n_banner += 1

    with open(COVERAGE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Final stale count
    rows2 = list(csv.DictReader(open(COVERAGE_CSV)))
    final_stale = sum(1 for r in rows2 if r["support_status"] == "composable" and has_stale((r.get("notes") or "").replace(BANNER, "")))
    print(f"Phrase rewrites:       {n_cleaned}")
    print(f"Banner-tagged rows:    {n_banner}")
    print(f"Composable rows still containing stale phrases (after banner):  {final_stale}")
    print(f"  (all such rows are now banner-tagged so readers know current state is composable)")


if __name__ == "__main__":
    main()
