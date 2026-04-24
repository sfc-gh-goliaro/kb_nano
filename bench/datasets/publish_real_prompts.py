#!/usr/bin/env python3
"""Publish the real-prompt benchmark workloads to the Hugging Face Hub.

Reads the JSON artefacts produced by
``python -m kb_nano.bench.datasets.build_real_prompts`` and uploads each
scenario as a standalone public dataset under
:data:`kb_nano.bench.datasets.HF_REPO_IDS` -- one repo per scenario:

  * ``sfc-gh-goliaro/kb-nano-prefill-heavy``
  * ``sfc-gh-goliaro/kb-nano-balanced``
  * ``sfc-gh-goliaro/kb-nano-decode-heavy``

Each repo contains:

  * ``data/train-00000-of-00001.parquet`` -- one row per request with fields
    ``messages`` (list[struct{role, content}]), ``assistant_text`` (str),
    ``source_id`` (str), ``oversized_at_build`` (bool). The chat content is
    stored as text so any model's tokenizer can be applied at benchmark time.
  * ``meta.json`` -- scenario-level metadata (build-time tokenizer, seed,
    stats, original cap configuration). Downloaded separately by the loader.
  * ``README.md`` -- human-readable dataset card.

Authentication: set ``HUGGINGFACE_HUB_TOKEN`` (or run ``huggingface-cli
login``) before invoking this script. The token must have write access to
the target org/user.

Usage::

    python -m kb_nano.bench.datasets.publish_real_prompts               # all
    python -m kb_nano.bench.datasets.publish_real_prompts \
        --scenario prefill-heavy
    python -m kb_nano.bench.datasets.publish_real_prompts --dry-run     # no upload
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

from kb_nano import KB_ROOT
from kb_nano.bench.datasets import HF_REPO_IDS, SCENARIO_NAMES

LOCAL_DIR = KB_ROOT / "bench" / "datasets" / "real_prompts"

# Keep parquet schema explicit so downstream `datasets.load_dataset` infers
# a stable Features object for every scenario.
_MESSAGE_STRUCT = pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
])

PARQUET_SCHEMA = pa.schema([
    pa.field("messages", pa.list_(_MESSAGE_STRUCT)),
    pa.field("assistant_text", pa.string()),
    pa.field("source_id", pa.string()),
    pa.field("oversized_at_build", pa.bool_()),
])


def _artifact_path(name: str) -> Path:
    path = LOCAL_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing local artefact for '{name}': {path}. Build it with "
            f"`python -m kb_nano.bench.datasets.build_real_prompts "
            f"--scenario {name}` first."
        )
    return path


def _to_parquet(artifact: dict, out_path: Path) -> None:
    requests = artifact["requests"]
    table = pa.table(
        {
            "messages": [
                [
                    {"role": str(m["role"]), "content": str(m["content"])}
                    for m in r["messages"]
                ]
                for r in requests
            ],
            "assistant_text": [str(r["assistant_text"]) for r in requests],
            "source_id": [str(r["source_id"]) for r in requests],
            "oversized_at_build": [
                bool(r["oversized_at_build"]) for r in requests
            ],
        },
        schema=PARQUET_SCHEMA,
    )
    pq.write_table(table, out_path, compression="zstd")


def _meta_json(artifact: dict) -> dict:
    return {
        "scenario": artifact["scenario"],
        "tokenizer": artifact["tokenizer"],
        "seed": artifact["seed"],
        "n_requests": artifact["n_requests"],
        "config": artifact["config"],
        "stats": artifact["stats"],
    }


def _render_readme(artifact: dict, repo_id: str) -> str:
    name = artifact["scenario"]
    cfg = artifact["config"]
    stats = artifact["stats"]
    ptok = stats["prompt_tokens"]
    dtok = stats["decode_tokens"]
    source = cfg.get("source", "?")
    dataset = cfg.get("dataset", "?")

    cap_lines = []
    if "prompt_cap" in cfg:
        cap_lines.append(
            f"- Prompt cap: **{cfg['prompt_cap']} tokens** (filtered at build "
            "time using the reference tokenizer; runner left-truncates any "
            "remaining oversized prompts to this cap)."
        )
    if "prompt_band" in cfg:
        lo, hi = cfg["prompt_band"]
        cap_lines.append(
            f"- Prompt band: **[{lo}, {hi}] tokens** (filtered at build time)."
        )
    cap_lines.append(
        f"- Decode cap: **{cfg['decode_cap']} tokens** (runner caps "
        "`min(len(tokenize(assistant_text)), decode_cap)` per request)."
    )
    if "decode_floor" in cfg:
        cap_lines.append(
            f"- Decode floor: **{cfg['decode_floor']} tokens** (build-time "
            "filter on the reference-tokenizer decode budget; the runtime "
            "decode for other tokenizers is not refloored, so a few requests "
            "may fall slightly below this when re-tokenized with a different "
            "model)."
        )
    caps = "\n".join(cap_lines)

    def _fmt(s):
        return (f"n={s['n']} · min={s['min']} · mean={s['mean']:.1f} · "
                f"median={s['median']} · p95={s['p95']} · p99={s['p99']} · "
                f"max={s['max']} · total={s['total']:,}")

    return f"""---
license: other
task_categories:
- text-generation
language:
- en
size_categories:
- n<1K
tags:
- benchmark
- llm-inference
- kb-nano
- {source}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# kb-nano `{name}` workload

Precomputed LLM throughput-benchmark workload used by
[`kb_nano`](https://github.com/sfc-gh-goliaro/kb_nano)'s `tests/bench_vllm.py`.

- **Source dataset:** `{dataset}` ({source})
- **Reference tokenizer (build-time, for filtering + stats):** `{artifact['tokenizer']}`
- **Seed:** `{artifact['seed']}`
- **Requests:** `{artifact['n_requests']}`
- **Oversized prompts at build:** `{stats['n_prompts_oversized']}`

## Caps applied at build time

{caps}

The chat content is stored as **raw text** so the benchmark runner can
re-tokenize the workload with **any model's tokenizer** (apply the chat
template + decode-cap at runtime) while preserving the same set of source
requests across models. Generation runs with `ignore_eos=True` and the
per-request decode budget = `min(len(tokenize(assistant_text)), decode_cap)`,
so wall-clock cost is deterministic for a given (model, scenario) pair.

## Token-length statistics (reference tokenizer)

| | stats |
|---|---|
| **prompt_tokens** | {_fmt(ptok)} |
| **decode_tokens** (capped) | {_fmt(dtok)} |

## Schema

Each row in `data/train-*.parquet`:

| field | type | description |
|---|---|---|
| `messages` | `list<struct<role: string, content: string>>` | Full chat history; the runner applies the target model's chat template with `add_generation_prompt=True`. |
| `assistant_text` | `string` | Natural assistant response from the source dataset. The runner tokenizes this with the target model's tokenizer and uses `min(len, decode_cap)` as the decode budget. |
| `source_id` | `string` | Provenance ID (task + example id or conversation hash). |
| `oversized_at_build` | `bool` | `true` if the prompt exceeded the cap when tokenized with the reference tokenizer; the runner is expected to left-truncate. |

Scenario-level metadata (stats, reference tokenizer, caps) is stored in
`meta.json` at the repo root.

## Loading

```python
from kb_nano.bench.utils.datasets import load_real_prompt_workload
wl = load_real_prompt_workload("{name}")
wl["messages"]         # list[list[{{role, content}}]]
wl["assistant_texts"]  # list[str]
wl["config"]           # {{prompt_cap | prompt_band, decode_cap, ...}}
```

Or directly:

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train")
```
"""


def publish_one(
    name: str,
    api: HfApi,
    dry_run: bool,
    workdir: Path,
) -> str:
    repo_id = HF_REPO_IDS[name]
    src = _artifact_path(name)
    with open(src) as f:
        artifact = json.load(f)
    if artifact["scenario"] != name:
        raise ValueError(
            f"artefact scenario mismatch: {artifact['scenario']!r} != {name!r}"
        )

    scenario_dir = workdir / name
    data_dir = scenario_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "train-00000-of-00001.parquet"
    meta_path = scenario_dir / "meta.json"
    readme_path = scenario_dir / "README.md"

    _to_parquet(artifact, parquet_path)
    with open(meta_path, "w") as f:
        json.dump(_meta_json(artifact), f, indent=2)
    readme_path.write_text(_render_readme(artifact, repo_id))

    size_mb = parquet_path.stat().st_size / 1e6
    print(
        f"[{name}] prepared repo content "
        f"(parquet={size_mb:.2f} MB, n={artifact['n_requests']}) -> {repo_id}"
    )

    if dry_run:
        print(f"  dry-run: skipping create_repo / upload for {repo_id}")
        return repo_id

    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(scenario_dir),
        commit_message=f"publish kb-nano {name} workload",
    )
    print(f"[{name}] uploaded -> https://huggingface.co/datasets/{repo_id}")
    return repo_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_NAMES) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build parquet + metadata locally but skip the Hub upload.",
    )
    args = parser.parse_args()

    names = list(SCENARIO_NAMES) if args.scenario == "all" else [args.scenario]
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="kb-nano-publish-") as td:
        workdir = Path(td)
        for name in names:
            publish_one(name, api, dry_run=args.dry_run, workdir=workdir)


if __name__ == "__main__":
    main()
