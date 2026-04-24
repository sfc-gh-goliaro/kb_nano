"""Thin adapter over vLLM's dataset infrastructure plus kb-nano helpers.

Re-exports the key types and functions from vllm.benchmarks.datasets so that
kb_nano benchmarks can use the exact same dataset CLI, loading, and sampling
logic -- including all dataset types (random, random-mm, sharegpt, sonnet,
burstgpt, hf, custom, custom_mm, prefix_repetition, etc.) and full multimodal
support.

Also exposes :func:`load_real_prompt_workload` for loading the real-prompt
LLM throughput workloads published to the Hugging Face Hub by
``kb_nano.bench.datasets.publish_real_prompts``. The workloads store raw
chat text, so callers can re-tokenize them with any target model's
tokenizer at benchmark time.
"""

from __future__ import annotations

import json

from vllm.benchmarks.datasets import (
    BenchmarkDataset,
    RandomDataset,
    RandomMultiModalDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    add_dataset_parser,
    add_random_dataset_base_args,
    add_random_multimodal_dataset_args,
    get_samples,
    process_image,
    process_video,
)

from kb_nano.bench.datasets import HF_REPO_IDS


def load_real_prompt_workload(name: str, revision: str | None = None) -> dict:
    """Download a real-prompt benchmark workload from the Hugging Face Hub.

    Fetches the parquet data + ``meta.json`` published by
    ``python -m kb_nano.bench.datasets.publish_real_prompts`` for the given
    scenario. Each request is returned as raw chat text (a list of
    ``{role, content}`` messages plus the source assistant response), so the
    caller can apply any target model's chat template + tokenizer.

    Parameters
    ----------
    name
        One of ``"prefill-heavy"``, ``"balanced"``, or ``"decode-heavy"``.
    revision
        Optional Hub revision (branch, tag, or commit hash). Defaults to the
        latest ``main``.

    Returns
    -------
    dict with keys:
        - ``scenario`` (str)
        - ``tokenizer`` (str)             -- build-time reference tokenizer
        - ``n_requests`` (int)
        - ``messages`` (list[list[dict]]) -- per-request chat history; each
          message is ``{"role": str, "content": str}``
        - ``assistant_texts`` (list[str]) -- per-request natural assistant
          response, used by the runner as a tokenizer-agnostic source for
          the decode budget (cap with ``config["decode_cap"]``)
        - ``source_ids`` (list[str])
        - ``oversized_at_build`` (list[bool])
        - ``stats`` (dict)                -- reference-tokenizer statistics
        - ``config`` (dict)               -- scenario's source/cap config
          (``prompt_cap`` or ``prompt_band``, ``decode_cap``, ...)
        - ``repo_id`` (str)               -- Hub repo the data was fetched from

    Raises
    ------
    KeyError
        If ``name`` is not a known scenario.
    """
    if name not in HF_REPO_IDS:
        raise KeyError(
            f"unknown real-prompt workload {name!r}; expected one of "
            f"{sorted(HF_REPO_IDS)}"
        )
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    repo_id = HF_REPO_IDS[name]

    ds = load_dataset(repo_id, split="train", revision=revision)

    meta_path = hf_hub_download(
        repo_id=repo_id,
        filename="meta.json",
        repo_type="dataset",
        revision=revision,
    )
    with open(meta_path) as f:
        meta = json.load(f)

    messages = [
        [{"role": str(m["role"]), "content": str(m["content"])} for m in row]
        for row in ds["messages"]
    ]
    assistant_texts = [str(x) for x in ds["assistant_text"]]
    source_ids = [str(x) for x in ds["source_id"]]
    oversized_at_build = [bool(x) for x in ds["oversized_at_build"]]

    return {
        "scenario": meta["scenario"],
        "tokenizer": meta["tokenizer"],
        "n_requests": int(meta.get("n_requests", len(messages))),
        "messages": messages,
        "assistant_texts": assistant_texts,
        "source_ids": source_ids,
        "oversized_at_build": oversized_at_build,
        "stats": meta["stats"],
        "config": meta["config"],
        "repo_id": repo_id,
    }


__all__ = [
    "BenchmarkDataset",
    "HF_REPO_IDS",
    "RandomDataset",
    "RandomMultiModalDataset",
    "SampleRequest",
    "ShareGPTDataset",
    "SonnetDataset",
    "add_dataset_parser",
    "add_random_dataset_base_args",
    "add_random_multimodal_dataset_args",
    "get_samples",
    "load_real_prompt_workload",
    "process_image",
    "process_video",
]
