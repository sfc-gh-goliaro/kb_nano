#!/usr/bin/env python3
"""Build Dr. Kernel validation Parquets from a pinned KernelBench checkout.

The released Level 2 dataset is the format and prompt-template authority. Before
writing anything, this script reconstructs all 100 Level 2 rows from the actual
KernelBench task files and requires byte-identical prompts and ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


KERNELBENCH_COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
EXPECTED_TASK_COUNTS = {1: 100, 2: 100, 3: 50}
OUTPUT_LEVELS = (1, 3)
TASK_NAME = re.compile(r"(?P<problem_id>\d+)_.+\.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kernelbench_repo",
        type=Path,
        help="KernelBench repository root (the directory containing KernelBench/)",
    )
    parser.add_argument(
        "released_level2",
        type=Path,
        help="Released Dr. Kernel validation_data_thinking.parquet",
    )
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def check_kernelbench_revision(repo: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if revision != KERNELBENCH_COMMIT:
        raise ValueError(
            f"KernelBench is at {revision}; expected {KERNELBENCH_COMMIT}"
        )


def task_files(repo: Path, level: int) -> list[tuple[int, Path]]:
    directory = repo / "KernelBench" / f"level{level}"
    paths = sorted(directory.glob("*.py"), key=lambda path: path.name)
    expected = EXPECTED_TASK_COUNTS[level]
    if len(paths) != expected:
        raise ValueError(f"Level {level} has {len(paths)} tasks; expected {expected}")

    tasks: list[tuple[int, Path]] = []
    seen_ids: set[int] = set()
    for path in paths:
        match = TASK_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected KernelBench task name: {path.name}")
        problem_id = int(match.group("problem_id"))
        if problem_id in seen_ids:
            raise ValueError(f"Duplicate Level {level} problem ID: {problem_id}")
        seen_ids.add(problem_id)
        tasks.append((problem_id, path))
    return tasks


def prompt_wrapper(released_rows: list[dict]) -> tuple[str, str]:
    row = released_rows[0]
    messages = row["prompt"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Released prompt is not a single user message")

    prompt = messages[0]["content"]
    reference = row["reward_model"]["ground_truth"]
    if prompt.count(reference) != 1:
        raise ValueError("Released prompt does not contain its reference exactly once")
    prefix, suffix = prompt.split(reference, 1)
    return prefix, suffix


def build_rows(repo: Path, level: int, prefix: str, suffix: str) -> list[dict]:
    rows = []
    for problem_id, path in task_files(repo, level):
        source = path.read_text(encoding="utf-8")
        rows.append(
            {
                "data_source": f"kernelbench_level{level}_validation",
                "prompt": [{"content": prefix + source + suffix, "role": "user"}],
                "reward_model": {"ground_truth": source, "style": "rule"},
                "ability": "kernel_optimization",
                "extra_info": {
                    "difficulty": None,
                    "name": path.stem,
                    "problem_id": problem_id,
                },
            }
        )
    return rows


def require_exact_level2(rebuilt: list[dict], released: list[dict]) -> None:
    if len(rebuilt) != len(released):
        raise ValueError(
            f"Rebuilt {len(rebuilt)} Level 2 rows; released dataset has {len(released)}"
        )

    for index, (actual, expected) in enumerate(zip(rebuilt, released, strict=True)):
        actual_prompt = actual["prompt"][0]["content"].encode("utf-8")
        expected_prompt = expected["prompt"][0]["content"].encode("utf-8")
        actual_reference = actual["reward_model"]["ground_truth"].encode("utf-8")
        expected_reference = expected["reward_model"]["ground_truth"].encode("utf-8")
        if actual_prompt != expected_prompt or actual_reference != expected_reference:
            problem_id = expected["extra_info"]["problem_id"]
            raise ValueError(
                f"Level 2 byte-identity gate failed at row {index}, problem {problem_id}"
            )
        if actual != expected:
            problem_id = expected["extra_info"]["problem_id"]
            raise ValueError(
                f"Level 2 metadata gate failed at row {index}, problem {problem_id}"
            )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    check_kernelbench_revision(args.kernelbench_repo)

    released_table = pq.read_table(args.released_level2)
    released_rows = released_table.to_pylist()
    prefix, suffix = prompt_wrapper(released_rows)

    rebuilt_level2 = build_rows(args.kernelbench_repo, 2, prefix, suffix)
    require_exact_level2(rebuilt_level2, released_rows)
    print("Level 2 gate passed: 100/100 rows are byte-identical")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for level in OUTPUT_LEVELS:
        rows = build_rows(args.kernelbench_repo, level, prefix, suffix)
        table = pa.Table.from_pylist(rows, schema=released_table.schema)
        output = args.output_dir / f"kernelbench_level{level}_validation.parquet"
        pq.write_table(table, output, compression="snappy", version="2.6")
        print(f"Wrote {len(rows)} rows to {output} (sha256={sha256(output)})")


if __name__ == "__main__":
    main()
