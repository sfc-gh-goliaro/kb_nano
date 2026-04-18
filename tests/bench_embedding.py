#!/usr/bin/env python3
"""
Throughput, latency, and alignment benchmark for embedding models.

Targets:
- BAAI/bge-m3
- colbert-ir/colbertv2.0

Backends:
- `FlagEmbedding` as the SOTA/reference baseline for BGE-M3
- official `ColBERT` model code as the SOTA/reference baseline for ColBERTv2
- repo-native `kb-nano` embedding model paths built from local tasks/modules

This intentionally does not use the current kb_nano generation engine because
that path is designed for autoregressive decoding rather than encoder-style
embedding workloads.

Usage:
    python tests/bench_embedding.py --model BAAI/bge-m3
    python tests/bench_embedding.py --model BAAI/bge-m3 --lengths 128,512
    python tests/bench_embedding.py --skip-reference
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def _detect_gpu_name() -> str:
    """Return short GPU name (for output directory naming)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _detect_embedding_family(model_name: str) -> str:
    basename = model_name.strip().lower().rsplit("/", 1)[-1]
    if basename in {"bge-m3", "bge_m3"}:
        return "bge_m3"
    if basename in {"colbertv2.0", "colbertv2", "colbert-v2", "colbert"}:
        return "colbertv2"
    raise ValueError(f"Unsupported embedding model for benchmark: {model_name}")


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _join_title_and_text(title: str | None, text: str | None) -> str:
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if text and text.strip():
        parts.append(text.strip())
    return " ".join(parts)


def _dataset_throughput_label(dataset_name: str, kind: str) -> str:
    return f"{dataset_name}-{kind}"


def _load_embedding_retrieval_dataset(
    dataset_name: str,
    cache_dir: str | None = None,
) -> dict:
    from datasets import load_dataset

    queries_ds = load_dataset(
        dataset_name,
        "queries",
        split="queries",
        cache_dir=cache_dir,
    )
    corpus_ds = load_dataset(
        dataset_name,
        "corpus",
        split="corpus",
        cache_dir=cache_dir,
    )
    qrels_ds = load_dataset(
        dataset_name,
        "default",
        split="test",
        cache_dir=cache_dir,
    )

    query_by_id = {}
    for item in queries_ds:
        text = (item.get("text") or "").strip()
        if text:
            query_by_id[str(item["_id"])] = text

    corpus_by_id = {}
    for item in corpus_ds:
        text = _join_title_and_text(item.get("title"), item.get("text"))
        if text:
            corpus_by_id[str(item["_id"])] = text

    pairs = []
    seen_pairs = set()
    for item in qrels_ds:
        if float(item.get("score", 0.0)) <= 0:
            continue
        query_id = str(item["query-id"])
        corpus_id = str(item["corpus-id"])
        if query_id not in query_by_id or corpus_id not in corpus_by_id:
            continue
        pair_key = (query_id, corpus_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        pairs.append({
            "query_id": query_id,
            "corpus_id": corpus_id,
            "query": query_by_id[query_id],
            "doc": corpus_by_id[corpus_id],
        })

    if not query_by_id:
        raise RuntimeError(f"No queries found in dataset {dataset_name}")
    if not corpus_by_id:
        raise RuntimeError(f"No corpus documents found in dataset {dataset_name}")
    if not pairs:
        raise RuntimeError(f"No positive query/doc pairs found in dataset {dataset_name}")

    return {
        "query_by_id": query_by_id,
        "corpus_by_id": corpus_by_id,
        "pairs": pairs,
    }


def _build_text_records(tokenizer, text_by_id: dict[str, str]) -> list[dict]:
    ids = list(text_by_id)
    texts = [text_by_id[item_id] for item_id in ids]
    lengths = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            verbose=False,
        )
        lengths.extend(len(item) for item in encoded["input_ids"])
    return [
        {"id": item_id, "text": text, "length": length}
        for item_id, text, length in zip(ids, texts, lengths, strict=True)
    ]


def _effective_model_max_length(tokenizer) -> int:
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None:
        return 512
    try:
        max_len = int(max_len)
    except (TypeError, ValueError):
        return 512
    if max_len <= 0 or max_len >= 100_000:
        return 512
    return max_len


def _sample_texts(
    records: list[dict],
    num_texts: int,
    seed: int,
    min_tokens: int | None = None,
) -> list[str]:
    if not records:
        raise RuntimeError("Cannot sample from an empty text pool")

    pool = records
    if min_tokens is not None:
        filtered = [item for item in records if item["length"] >= min_tokens]
        if filtered:
            pool = filtered
        else:
            pool = sorted(records, key=lambda item: item["length"], reverse=True)
            pool = pool[:max(1, min(num_texts, len(pool)))]

    rng = random.Random(seed)
    if len(pool) >= num_texts:
        chosen = rng.sample(pool, k=num_texts)
        return [item["text"] for item in chosen]

    chosen = list(pool)
    rng.shuffle(chosen)
    out = [item["text"] for item in chosen]
    while len(out) < num_texts:
        out.append(rng.choice(pool)["text"])
    return out


def _sample_pairs(
    pair_records: list[dict],
    num_pairs: int,
    seed: int,
    min_query_tokens: int | None = None,
    min_doc_tokens: int | None = None,
) -> list[dict]:
    if not pair_records:
        raise RuntimeError("Cannot sample from an empty pair pool")

    pool = pair_records
    if min_query_tokens is not None or min_doc_tokens is not None:
        filtered = [
            item
            for item in pair_records
            if (min_query_tokens is None or item["query_length"] >= min_query_tokens)
            and (min_doc_tokens is None or item["doc_length"] >= min_doc_tokens)
        ]
        if filtered:
            pool = filtered
        else:
            pool = sorted(
                pair_records,
                key=lambda item: (item["doc_length"], item["query_length"]),
                reverse=True,
            )
            pool = pool[:max(1, min(num_pairs, len(pool)))]

    rng = random.Random(seed)
    if len(pool) >= num_pairs:
        return rng.sample(pool, k=num_pairs)

    chosen = list(pool)
    rng.shuffle(chosen)
    while len(chosen) < num_pairs:
        chosen.append(rng.choice(pool))
    return chosen


def _prepare_bge_dataset_workload(
    model_name: str,
    dataset_name: str,
    dataset_cache_dir: str | None,
    lengths: list[int],
    latency_batch_sizes: list[int],
    args,
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    native_doc_max_len = _effective_model_max_length(tokenizer)
    dataset = _load_embedding_retrieval_dataset(dataset_name, cache_dir=dataset_cache_dir)
    corpus_records = _build_text_records(tokenizer, dataset["corpus_by_id"])

    scenarios = []
    if not args.skip_throughput:
        scenarios.append({
            "name": _dataset_throughput_label(dataset_name, "doc"),
            "target_label": "dataset",
            "target_len": native_doc_max_len,
            "num_texts": args.num_texts,
            "texts": _sample_texts(
                corpus_records,
                args.num_texts,
                seed=args.seed + 11,
            ),
        })
        for idx, seq_len in enumerate(lengths):
            scenarios.append({
                "name": f"len-{seq_len}",
                "target_label": str(seq_len),
                "target_len": seq_len,
                "num_texts": args.num_texts,
                "texts": _sample_texts(
                    corpus_records,
                    args.num_texts,
                    seed=args.seed + 101 * (idx + 1) + seq_len,
                    min_tokens=seq_len,
                ),
            })

    latency_scenarios = []
    if not args.skip_latency:
        for idx, bs in enumerate(latency_batch_sizes):
            latency_scenarios.append({
                "name": f"bs-{bs}-len-{args.latency_len}",
                "batch_size": bs,
                "target_len": args.latency_len,
                "num_warmup": 2,
                "num_iters": args.latency_iters,
                "texts": _sample_texts(
                    corpus_records,
                    bs,
                    seed=args.seed + 701 + idx,
                    min_tokens=args.latency_len,
                ),
            })

    warm_texts = _sample_texts(
        corpus_records,
        min(4, args.batch_size),
        seed=args.seed + 17,
        min_tokens=32,
    )

    alignment_texts = []
    if not args.skip_alignment:
        alignment_texts = _sample_texts(
            corpus_records,
            args.alignment_texts,
            seed=args.seed + 500,
            min_tokens=args.alignment_len,
        )

    return {
        "dataset_name": dataset_name,
        "num_queries": len(dataset["query_by_id"]),
        "num_docs": len(dataset["corpus_by_id"]),
        "num_pairs": len(dataset["pairs"]),
        "native_doc_max_len": native_doc_max_len,
        "warm_texts": warm_texts,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "alignment_texts": alignment_texts,
    }


def _prepare_colbert_dataset_workload(
    model_name: str,
    dataset_name: str,
    dataset_cache_dir: str | None,
    lengths: list[int],
    latency_batch_sizes: list[int],
    args,
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    native_doc_max_len = _effective_model_max_length(tokenizer)
    dataset = _load_embedding_retrieval_dataset(dataset_name, cache_dir=dataset_cache_dir)
    query_records = _build_text_records(tokenizer, dataset["query_by_id"])
    doc_records = _build_text_records(tokenizer, dataset["corpus_by_id"])

    query_length_by_id = {item["id"]: item["length"] for item in query_records}
    doc_length_by_id = {item["id"]: item["length"] for item in doc_records}
    pair_records = [
        {
            **item,
            "query_length": query_length_by_id[item["query_id"]],
            "doc_length": doc_length_by_id[item["corpus_id"]],
        }
        for item in dataset["pairs"]
    ]

    scenarios = []
    if not args.skip_throughput:
        scenarios.append({
            "name": _dataset_throughput_label(dataset_name, "query"),
            "mode": "query",
            "target_label": "dataset",
            "target_len": args.query_len,
            "num_texts": args.num_texts,
            "texts": _sample_texts(
                query_records,
                args.num_texts,
                seed=args.seed + 101,
            ),
        })
        scenarios.append({
            "name": _dataset_throughput_label(dataset_name, "doc"),
            "mode": "doc",
            "target_label": "dataset",
            "target_len": native_doc_max_len,
            "num_texts": args.num_texts,
            "texts": _sample_texts(
                doc_records,
                args.num_texts,
                seed=args.seed + 202,
            ),
        })
        for idx, seq_len in enumerate(lengths):
            scenarios.append({
                "name": f"doc-len-{seq_len}",
                "mode": "doc",
                "target_label": str(seq_len),
                "target_len": seq_len,
                "num_texts": args.num_texts,
                "texts": _sample_texts(
                    doc_records,
                    args.num_texts,
                    seed=args.seed + 303 + idx + seq_len,
                    min_tokens=seq_len,
                ),
            })

    latency_scenarios = []
    if not args.skip_latency:
        for idx, bs in enumerate(latency_batch_sizes):
            latency_scenarios.append({
                "name": f"query-bs-{bs}-len-{args.query_len}",
                "mode": "query",
                "batch_size": bs,
                "target_len": args.query_len,
                "num_warmup": 2,
                "num_iters": args.latency_iters,
                "texts": _sample_texts(
                    query_records,
                    bs,
                    seed=args.seed + 901 + idx,
                ),
            })
        for idx, bs in enumerate(latency_batch_sizes):
            latency_scenarios.append({
                "name": f"doc-bs-{bs}-len-{args.latency_len}",
                "mode": "doc",
                "batch_size": bs,
                "target_len": args.latency_len,
                "num_warmup": 2,
                "num_iters": args.latency_iters,
                "texts": _sample_texts(
                    doc_records,
                    bs,
                    seed=args.seed + 1201 + idx,
                    min_tokens=args.latency_len,
                ),
            })

    warm_queries = _sample_texts(
        query_records,
        min(4, args.batch_size),
        seed=args.seed + 111,
    )
    warm_docs = _sample_texts(
        doc_records,
        min(4, args.batch_size),
        seed=args.seed + 222,
        min_tokens=32,
    )

    alignment_queries = []
    alignment_docs = []
    if not args.skip_alignment:
        sampled_pairs = _sample_pairs(
            pair_records,
            args.alignment_texts,
            seed=args.seed + 500,
            min_doc_tokens=args.alignment_len,
        )
        alignment_queries = [item["query"] for item in sampled_pairs]
        alignment_docs = [item["doc"] for item in sampled_pairs]

    return {
        "dataset_name": dataset_name,
        "num_queries": len(dataset["query_by_id"]),
        "num_docs": len(dataset["corpus_by_id"]),
        "num_pairs": len(pair_records),
        "native_doc_max_len": native_doc_max_len,
        "warm_queries": warm_queries,
        "warm_docs": warm_docs,
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "alignment_queries": alignment_queries,
        "alignment_docs": alignment_docs,
    }


def _compute_alignment(local_outputs: dict, ref_outputs: dict) -> dict:
    """Compute dense/sparse alignment metrics."""
    result: dict[str, dict] = {}

    local_dense = local_outputs.get("dense_vecs")
    ref_dense = ref_outputs.get("dense_vecs")
    if local_dense is not None and ref_dense is not None:
        a = np.asarray(local_dense, dtype=np.float32)
        b = np.asarray(ref_dense, dtype=np.float32)
        if a.shape != b.shape:
            result["dense"] = {
                "shape_match": False,
                "local_shape": list(a.shape),
                "reference_shape": list(b.shape),
            }
        else:
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            cosine = np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)
            abs_diff = np.abs(a - b)
            result["dense"] = {
                "shape_match": True,
                "avg_cosine_similarity": float(np.mean(cosine)),
                "min_cosine_similarity": float(np.min(cosine)),
                "avg_mean_abs_diff": float(np.mean(np.mean(abs_diff, axis=1))),
                "max_abs_diff": float(np.max(abs_diff)),
            }

    local_sparse = local_outputs.get("lexical_weights")
    ref_sparse = ref_outputs.get("lexical_weights")
    if local_sparse is not None and ref_sparse is not None:
        key_jaccards = []
        exact_key_matches = 0
        mean_abs_diffs = []
        max_abs_diffs = []
        for local_item, ref_item in zip(local_sparse, ref_sparse):
            local_item = {str(k): float(v) for k, v in local_item.items()}
            ref_item = {str(k): float(v) for k, v in ref_item.items()}
            local_keys = set(local_item)
            ref_keys = set(ref_item)
            union = local_keys | ref_keys
            inter = local_keys & ref_keys
            if union:
                key_jaccards.append(len(inter) / len(union))
                diffs = [abs(local_item.get(k, 0.0) - ref_item.get(k, 0.0)) for k in union]
                mean_abs_diffs.append(float(np.mean(diffs)))
                max_abs_diffs.append(float(np.max(diffs)))
            else:
                key_jaccards.append(1.0)
                mean_abs_diffs.append(0.0)
                max_abs_diffs.append(0.0)
            if local_keys == ref_keys:
                exact_key_matches += 1

        total = len(local_sparse)
        result["sparse"] = {
            "avg_key_jaccard": float(np.mean(key_jaccards)) if key_jaccards else 0.0,
            "exact_key_match_rate": _safe_div(exact_key_matches, total),
            "avg_mean_abs_diff": float(np.mean(mean_abs_diffs)) if mean_abs_diffs else 0.0,
            "max_abs_diff": float(np.max(max_abs_diffs)) if max_abs_diffs else 0.0,
        }

    return result


def _compute_colbert_vec_alignment(local_vecs: list, ref_vecs: list) -> dict:
    if len(local_vecs) != len(ref_vecs):
        return {
            "shape_match": False,
            "local_count": len(local_vecs),
            "reference_count": len(ref_vecs),
        }

    local_rows = []
    ref_rows = []
    for local_item, ref_item in zip(local_vecs, ref_vecs):
        a = np.asarray(local_item, dtype=np.float32)
        b = np.asarray(ref_item, dtype=np.float32)
        if a.shape != b.shape:
            return {
                "shape_match": False,
                "local_shape": list(a.shape),
                "reference_shape": list(b.shape),
            }
        if a.size == 0:
            continue
        local_rows.append(a.reshape(-1, a.shape[-1]))
        ref_rows.append(b.reshape(-1, b.shape[-1]))

    if not local_rows:
        return {
            "shape_match": True,
            "avg_cosine_similarity": 1.0,
            "min_cosine_similarity": 1.0,
            "avg_mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
        }

    a = np.concatenate(local_rows, axis=0)
    b = np.concatenate(ref_rows, axis=0)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)
    abs_diff = np.abs(a - b)
    return {
        "shape_match": True,
        "avg_cosine_similarity": float(np.mean(cosine)),
        "min_cosine_similarity": float(np.min(cosine)),
        "avg_mean_abs_diff": float(np.mean(np.mean(abs_diff, axis=1))),
        "max_abs_diff": float(np.max(abs_diff)),
    }


def _compute_colbert_alignment(local_outputs: dict, ref_outputs: dict) -> dict:
    result: dict[str, dict] = {}

    query_vecs = local_outputs.get("query_vecs")
    ref_query_vecs = ref_outputs.get("query_vecs")
    if query_vecs is not None and ref_query_vecs is not None:
        result["query"] = _compute_colbert_vec_alignment(query_vecs, ref_query_vecs)

    doc_vecs = local_outputs.get("doc_vecs")
    ref_doc_vecs = ref_outputs.get("doc_vecs")
    if doc_vecs is not None and ref_doc_vecs is not None:
        result["doc"] = _compute_colbert_vec_alignment(doc_vecs, ref_doc_vecs)

    local_scores = local_outputs.get("scores")
    ref_scores = ref_outputs.get("scores")
    if local_scores is not None and ref_scores is not None:
        a = np.asarray(local_scores, dtype=np.float32)
        b = np.asarray(ref_scores, dtype=np.float32)
        if a.shape != b.shape:
            result["scores"] = {
                "shape_match": False,
                "local_shape": list(a.shape),
                "reference_shape": list(b.shape),
            }
        else:
            abs_diff = np.abs(a - b)
            result["scores"] = {
                "shape_match": True,
                "avg_mean_abs_diff": float(np.mean(abs_diff)),
                "max_abs_diff": float(np.max(abs_diff)),
            }

    return result


_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent

_WORKER_PATH = _PACKAGE_DIR / "bench" / "utils" / "worker.py"
_WORKER_SPEC = importlib.util.spec_from_file_location("bench_worker_local", _WORKER_PATH)
if _WORKER_SPEC is None or _WORKER_SPEC.loader is None:
    raise RuntimeError(f"Could not load worker helper from {_WORKER_PATH}")
_WORKER_MODULE = importlib.util.module_from_spec(_WORKER_SPEC)
_WORKER_SPEC.loader.exec_module(_WORKER_MODULE)
run_worker = _WORKER_MODULE.run_worker


_EMBED_WORKER_SHARED = r'''
import json
import random
import sys
import time

import numpy as np
import torch

WORDS = [
    "retrieval", "embedding", "sparse", "dense", "document", "query",
    "passage", "semantic", "lexical", "ranking", "benchmark", "system",
    "search", "knowledge", "encoder", "vector", "token", "multilingual",
    "hybrid", "matching", "index", "context", "pipeline", "evaluation",
    "dataset", "throughput", "latency", "precision", "feature", "signal",
    "corpus", "kb", "nano", "baseline", "model", "alignment",
]


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_texts(tokenizer, num_texts, target_len, seed):
    texts = []
    token_counts = []
    rng = random.Random(seed + 7919 * target_len)
    for i in range(num_texts):
        parts = [f"sample{i}", "retrieval", "embedding"]
        while True:
            parts.extend(rng.sample(WORDS, k=min(4, len(WORDS))))
            text = " ".join(parts)
            count = len(
                tokenizer(
                    text,
                    truncation=True,
                    max_length=target_len,
                )["input_ids"]
            )
            if count >= target_len:
                texts.append(text)
                token_counts.append(target_len)
                break
    return texts, token_counts


def _token_counts_for_texts(tokenizer, texts, max_length):
    return [
        len(
            tokenizer(
                text,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
        )
        for text in texts
    ]


def _sanitize_outputs(outputs):
    sanitized = {}
    dense = outputs.get("dense_vecs")
    if dense is not None:
        sanitized["dense_vecs"] = np.asarray(dense, dtype=np.float32).tolist()

    lexical = outputs.get("lexical_weights")
    if lexical is not None:
        sanitized["lexical_weights"] = [
            {str(k): float(v) for k, v in item.items()}
            for item in lexical
        ]

    colbert = outputs.get("colbert_vecs")
    if colbert is not None:
        sanitized["colbert_vecs"] = [
            np.asarray(item, dtype=np.float32).tolist()
            for item in colbert
        ]

    return sanitized


def _run_suite(backend, cfg):
    scenarios = cfg["scenarios"]
    latency_specs = cfg.get("latency_scenarios", [])
    return_dense = cfg.get("return_dense", True)
    return_sparse = cfg.get("return_sparse", False)
    return_colbert_vecs = cfg.get("return_colbert_vecs", False)
    batch_size = cfg["batch_size"]

    warm_len = scenarios[0]["target_len"] if scenarios else cfg.get("alignment_target_len", 32)
    warm_texts = list(cfg.get("warm_texts") or [])
    if not warm_texts and scenarios and scenarios[0].get("texts"):
        warm_texts = list(scenarios[0]["texts"][:min(4, batch_size)])
    if not warm_texts and latency_specs and latency_specs[0].get("texts"):
        warm_texts = list(latency_specs[0]["texts"][:min(4, batch_size)])
    if not warm_texts and cfg.get("alignment_texts"):
        warm_texts = list(cfg["alignment_texts"][:min(4, batch_size)])
    if not warm_texts:
        warm_texts, _ = _make_texts(
            backend.tokenizer,
            min(4, batch_size),
            min(32, warm_len),
            cfg["seed"] + 999,
        )
    backend.encode(
        warm_texts,
        batch_size=min(batch_size, len(warm_texts)),
        max_length=min(32, warm_len),
        return_dense=return_dense,
        return_sparse=return_sparse,
        return_colbert_vecs=return_colbert_vecs,
    )

    throughput = []
    for idx, scenario in enumerate(scenarios):
        if scenario.get("texts"):
            texts = list(scenario["texts"])
            token_counts = _token_counts_for_texts(
                backend.tokenizer,
                texts,
                scenario["target_len"],
            )
        else:
            texts, token_counts = _make_texts(
                backend.tokenizer,
                scenario["num_texts"],
                scenario["target_len"],
                cfg["seed"] + idx,
            )
        _cuda_sync()
        start = time.perf_counter()
        backend.encode(
            texts,
            batch_size=scenario.get("batch_size", batch_size),
            max_length=scenario["target_len"],
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
        )
        _cuda_sync()
        elapsed = time.perf_counter() - start
        throughput.append({
            "name": scenario["name"],
            "target_label": scenario.get("target_label"),
            "num_texts": len(texts),
            "target_len": scenario["target_len"],
            "total_input_tokens": int(sum(token_counts)),
            "elapsed": elapsed,
        })

    latency = []
    for idx, spec in enumerate(latency_specs):
        if spec.get("texts"):
            texts = list(spec["texts"])
            token_counts = _token_counts_for_texts(
                backend.tokenizer,
                texts,
                spec["target_len"],
            )
        else:
            texts, token_counts = _make_texts(
                backend.tokenizer,
                spec["batch_size"],
                spec["target_len"],
                cfg["seed"] + 200 + idx,
            )
        run_kwargs = dict(
            batch_size=spec["batch_size"],
            max_length=spec["target_len"],
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
        )
        for _ in range(spec.get("num_warmup", 2)):
            backend.encode(texts, **run_kwargs)
        latencies = []
        for _ in range(spec.get("num_iters", 5)):
            _cuda_sync()
            start = time.perf_counter()
            backend.encode(texts, **run_kwargs)
            _cuda_sync()
            latencies.append(time.perf_counter() - start)
        latency.append({
            "name": spec["name"],
            "batch_size": spec["batch_size"],
            "target_len": spec["target_len"],
            "total_input_tokens": int(sum(token_counts)),
            "latencies": latencies,
            "num_iters": spec.get("num_iters", 5),
        })

    alignment = None
    if cfg.get("alignment_num_texts", 0) > 0:
        if cfg.get("alignment_texts"):
            texts = list(cfg["alignment_texts"])
            token_counts = _token_counts_for_texts(
                backend.tokenizer,
                texts,
                cfg["alignment_target_len"],
            )
        else:
            texts, token_counts = _make_texts(
                backend.tokenizer,
                cfg["alignment_num_texts"],
                cfg["alignment_target_len"],
                cfg["seed"] + 500,
            )
        outputs = backend.encode(
            texts,
            batch_size=min(batch_size, len(texts)),
            max_length=cfg["alignment_target_len"],
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
        )
        alignment = {
            "num_texts": len(texts),
            "target_len": cfg["alignment_target_len"],
            "total_input_tokens": int(sum(token_counts)),
            "outputs": _sanitize_outputs(outputs),
        }

    return {
        "throughput": throughput,
        "latency": latency,
        "alignment": alignment,
    }
'''


KB_BGEM3_WORKER = _EMBED_WORKER_SHARED + r'''
import importlib.util
import os


def _load_local_package(repo_dir):
    init_path = os.path.join(repo_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "kb_nano_local",
        init_path,
        submodule_search_locations=[repo_dir],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load local package from {repo_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kb_nano_local"] = module
    spec.loader.exec_module(module)
    return "kb_nano_local"


class KBBGEM3Backend:
    def __init__(self, cfg):
        from transformers import AutoTokenizer

        pkg_name = _load_local_package(cfg["repo_dir"])
        embedder_mod = __import__(
            f"{pkg_name}.infra.embedder_loader",
            fromlist=["load_bge_m3_model"],
        )
        self.model_name = cfg["model"]
        self.device = cfg.get("device", "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model, self.config = embedder_mod.load_bge_m3_model(
            self.model_name,
            device=torch.device(self.device),
            dtype=torch.float16 if cfg.get("use_fp16", False) and self.device.startswith("cuda") else torch.float32,
        )

        self.unused_token_ids = {
            tid for tid in (
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            )
            if tid is not None
        }

    def _to_device(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _process_token_weights(self, token_weights, input_ids):
        result = {}
        for weight, idx in zip(token_weights, input_ids):
            idx = int(idx)
            if idx in self.unused_token_ids or weight <= 0:
                continue
            key = str(idx)
            value = float(weight)
            if value > result.get(key, float("-inf")):
                result[key] = value
        return result

    def encode(self, texts, batch_size, max_length,
               return_dense=True, return_sparse=False,
               return_colbert_vecs=False):
        dense_batches = []
        lexical_weights = []
        colbert_vecs = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = self._to_device(inputs)

            with torch.inference_mode():
                outputs = self.model(
                    text_input=inputs,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=return_colbert_vecs,
                    return_sparse_embedding=False,
                )

                if return_dense:
                    dense_batches.append(outputs["dense_vecs"].float().cpu().numpy())

                if return_sparse:
                    token_weights = outputs["sparse_vecs"].squeeze(-1)
                    token_weights_cpu = token_weights.float().cpu().numpy()
                    input_ids_cpu = inputs["input_ids"].cpu().numpy()
                    for weights, ids in zip(token_weights_cpu, input_ids_cpu):
                        lexical_weights.append(self._process_token_weights(weights, ids))

                if return_colbert_vecs:
                    colbert_cpu = outputs["colbert_vecs"].float().cpu().numpy()
                    mask_cpu = inputs["attention_mask"].cpu().numpy()
                    for item, mask in zip(colbert_cpu, mask_cpu):
                        num_tokens = int(np.sum(mask))
                        colbert_vecs.append(item[:max(num_tokens - 1, 0)])

        results = {}
        if return_dense:
            results["dense_vecs"] = (
                np.concatenate(dense_batches, axis=0)
                if dense_batches
                else np.empty((0, self.config.hidden_size), dtype=np.float32)
            )
        if return_sparse:
            results["lexical_weights"] = lexical_weights
        if return_colbert_vecs:
            results["colbert_vecs"] = colbert_vecs
        return results


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    backend = KBBGEM3Backend(cfg)
    results = _run_suite(backend, cfg)
    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
'''


FLAGEMBEDDING_WORKER = _EMBED_WORKER_SHARED + r'''
import torch.nn.functional as F


class LightweightBGEM3FlagModel:
    def __init__(
        self,
        model_name_or_path,
        use_fp16=True,
        devices="cpu",
        pooling_method="cls",
        normalize_embeddings=True,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    ):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name_or_path
        self.device = devices if isinstance(devices, str) else devices[0]
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.return_dense = return_dense
        self.return_sparse = return_sparse
        self.return_colbert_vecs = return_colbert_vecs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        use_half = use_fp16 and str(self.device).startswith("cuda")
        model_dtype = torch.float16 if use_half else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=False,
        )
        self.model.to(self.device)
        self.model.eval()

        hidden_size = self.model.config.hidden_size
        self.sparse_linear = torch.nn.Linear(hidden_size, 1, bias=True)
        self.colbert_linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.sparse_linear.load_state_dict(
            torch.load(
                hf_hub_download(model_name_or_path, "sparse_linear.pt"),
                map_location="cpu",
            ),
        )
        self.colbert_linear.load_state_dict(
            torch.load(
                hf_hub_download(model_name_or_path, "colbert_linear.pt"),
                map_location="cpu",
            ),
        )
        self.sparse_linear.to(self.device, dtype=model_dtype)
        self.colbert_linear.to(self.device, dtype=model_dtype)
        self.sparse_linear.eval()
        self.colbert_linear.eval()

        self.unused_token_ids = {
            tid for tid in (
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            )
            if tid is not None
        }

    def _to_device(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _dense_embedding(self, last_hidden_state, attention_mask):
        if self.pooling_method == "cls":
            return last_hidden_state[:, 0]
        if self.pooling_method == "mean":
            summed = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            denom = attention_mask.sum(dim=1, keepdim=True).float()
            return summed / denom
        raise NotImplementedError(f"Unsupported pooling method: {self.pooling_method}")

    def _process_token_weights(self, token_weights, input_ids):
        result = {}
        for weight, idx in zip(token_weights, input_ids):
            idx = int(idx)
            if idx in self.unused_token_ids or weight <= 0:
                continue
            key = str(idx)
            value = float(weight)
            if value > result.get(key, float("-inf")):
                result[key] = value
        return result

    def encode(
        self,
        texts,
        batch_size,
        max_length,
        return_dense=None,
        return_sparse=None,
        return_colbert_vecs=None,
    ):
        if return_dense is None:
            return_dense = self.return_dense
        if return_sparse is None:
            return_sparse = self.return_sparse
        if return_colbert_vecs is None:
            return_colbert_vecs = self.return_colbert_vecs

        dense_batches = []
        lexical_weights = []
        colbert_vecs = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = self._to_device(inputs)

            with torch.inference_mode():
                last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state

                if return_dense:
                    dense = self._dense_embedding(last_hidden_state, inputs["attention_mask"])
                    if self.normalize_embeddings:
                        dense = F.normalize(dense, dim=-1)
                    dense_batches.append(dense.float().cpu().numpy())

                if return_sparse:
                    token_weights = torch.relu(self.sparse_linear(last_hidden_state)).squeeze(-1)
                    token_weights_cpu = token_weights.float().cpu().numpy()
                    input_ids_cpu = inputs["input_ids"].cpu().numpy()
                    for weights, ids in zip(token_weights_cpu, input_ids_cpu):
                        lexical_weights.append(self._process_token_weights(weights, ids))

                if return_colbert_vecs:
                    colbert = self.colbert_linear(last_hidden_state[:, 1:])
                    colbert = colbert * inputs["attention_mask"][:, 1:][:, :, None].float()
                    if self.normalize_embeddings:
                        colbert = F.normalize(colbert, dim=-1)
                    colbert_cpu = colbert.float().cpu().numpy()
                    mask_cpu = inputs["attention_mask"].cpu().numpy()
                    for item, mask in zip(colbert_cpu, mask_cpu):
                        num_tokens = int(np.sum(mask))
                        colbert_vecs.append(item[:max(num_tokens - 1, 0)])

        results = {}
        if return_dense:
            results["dense_vecs"] = np.concatenate(dense_batches, axis=0)
        if return_sparse:
            results["lexical_weights"] = lexical_weights
        if return_colbert_vecs:
            results["colbert_vecs"] = colbert_vecs
        return results


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    try:
        from FlagEmbedding import BGEM3FlagModel
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise
        print(
            "WARNING: FlagEmbedding import requires optional dependency 'datasets'; "
            "falling back to lightweight official M3 inference path.",
            file=sys.stderr,
        )
        BGEM3FlagModel = LightweightBGEM3FlagModel

    backend = BGEM3FlagModel(
        model_name_or_path=cfg["model"],
        use_fp16=cfg.get("use_fp16", True),
        devices=cfg.get("device", "cpu"),
        pooling_method="cls",
        normalize_embeddings=cfg.get("normalize_embeddings", True),
        return_dense=cfg.get("return_dense", True),
        return_sparse=cfg.get("return_sparse", False),
        return_colbert_vecs=cfg.get("return_colbert_vecs", False),
    )
    results = _run_suite(backend, cfg)
    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
'''


_COLBERT_WORKER_SHARED = r'''
import json
import random
import string
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

WORDS = [
    "retrieval", "embedding", "sparse", "dense", "document", "query",
    "passage", "semantic", "lexical", "ranking", "benchmark", "system",
    "search", "knowledge", "encoder", "vector", "token", "multilingual",
    "hybrid", "matching", "index", "context", "pipeline", "evaluation",
    "dataset", "throughput", "latency", "precision", "feature", "signal",
    "corpus", "kb", "nano", "baseline", "model", "alignment",
]


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _insert_prefix_token(tensor, prefix_id):
    prefix = torch.full(
        (tensor.size(0), 1),
        prefix_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor[:, :1], prefix, tensor[:, 1:]], dim=1)


def _make_texts(tokenizer, num_texts, target_len, seed):
    texts = []
    rng = random.Random(seed + 3571 * target_len)
    for i in range(num_texts):
        parts = [f"sample{i}", "retrieval", "search"]
        while True:
            parts.extend(rng.sample(WORDS, k=min(4, len(WORDS))))
            text = " ".join(parts)
            count = len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=target_len,
                )["input_ids"]
            )
            if count >= target_len:
                texts.append(text)
                break
    return texts


def _tensorize_queries(
    tokenizer,
    texts,
    max_length,
    query_marker_token_id,
    mask_token_id,
    attend_to_mask_tokens=False,
):
    obj = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length - 1,
    )
    input_ids = _insert_prefix_token(obj["input_ids"], query_marker_token_id)
    attention_mask = _insert_prefix_token(obj["attention_mask"], 1)
    input_ids[input_ids == tokenizer.pad_token_id] = mask_token_id
    if attend_to_mask_tokens:
        attention_mask[input_ids == mask_token_id] = 1
    return input_ids, attention_mask


def _tensorize_docs(tokenizer, texts, max_length, doc_marker_token_id):
    obj = tokenizer(
        texts,
        padding="longest",
        truncation="longest_first",
        return_tensors="pt",
        max_length=max_length - 1,
    )
    input_ids = _insert_prefix_token(obj["input_ids"], doc_marker_token_id)
    attention_mask = _insert_prefix_token(obj["attention_mask"], 1)
    return input_ids, attention_mask


def _sanitize_vecs(vecs):
    return [np.asarray(item, dtype=np.float32).tolist() for item in vecs]


def _run_colbert_suite(backend, cfg):
    scenarios = cfg["scenarios"]
    latency_specs = cfg.get("latency_scenarios", [])
    batch_size = cfg["batch_size"]

    warm_queries = list(cfg.get("warm_queries") or [])
    if not warm_queries and scenarios:
        for scenario in scenarios:
            if scenario.get("mode") == "query" and scenario.get("texts"):
                warm_queries = list(scenario["texts"][:min(4, batch_size)])
                break
    if not warm_queries:
        warm_queries = _make_texts(
            backend.tokenizer,
            min(4, batch_size),
            cfg.get("query_len", 32),
            cfg["seed"] + 111,
        )

    warm_docs = list(cfg.get("warm_docs") or [])
    if not warm_docs and scenarios:
        for scenario in scenarios:
            if scenario.get("mode") == "doc" and scenario.get("texts"):
                warm_docs = list(scenario["texts"][:min(4, batch_size)])
                break
    if not warm_docs:
        warm_docs = _make_texts(
            backend.tokenizer,
            min(4, batch_size),
            min(32, cfg.get("doc_len", 128)),
            cfg["seed"] + 222,
        )
    backend.encode(
        warm_queries,
        mode="query",
        batch_size=min(batch_size, len(warm_queries)),
        max_length=cfg.get("query_len", 32),
    )
    backend.encode(
        warm_docs,
        mode="doc",
        batch_size=min(batch_size, len(warm_docs)),
        max_length=min(32, cfg.get("doc_len", 128)),
    )

    throughput = []
    for idx, scenario in enumerate(scenarios):
        if scenario.get("texts"):
            texts = list(scenario["texts"])
        else:
            texts = _make_texts(
                backend.tokenizer,
                scenario["num_texts"],
                scenario["target_len"],
                cfg["seed"] + idx,
            )
        _cuda_sync()
        start = time.perf_counter()
        outputs = backend.encode(
            texts,
            mode=scenario["mode"],
            batch_size=scenario.get("batch_size", batch_size),
            max_length=scenario["target_len"],
        )
        _cuda_sync()
        throughput.append({
            "name": scenario["name"],
            "mode": scenario["mode"],
            "target_label": scenario.get("target_label"),
            "num_texts": len(texts),
            "target_len": scenario["target_len"],
            "total_input_tokens": int(outputs["total_input_tokens"]),
            "elapsed": time.perf_counter() - start,
        })

    latency = []
    for idx, spec in enumerate(latency_specs):
        if spec.get("texts"):
            texts = list(spec["texts"])
        else:
            texts = _make_texts(
                backend.tokenizer,
                spec["batch_size"],
                spec["target_len"],
                cfg["seed"] + 200 + idx,
            )
        run_kwargs = dict(
            mode=spec["mode"],
            batch_size=spec["batch_size"],
            max_length=spec["target_len"],
        )
        for _ in range(spec.get("num_warmup", 2)):
            backend.encode(texts, **run_kwargs)
        latencies = []
        total_input_tokens = None
        for _ in range(spec.get("num_iters", 5)):
            _cuda_sync()
            start = time.perf_counter()
            outputs = backend.encode(texts, **run_kwargs)
            _cuda_sync()
            latencies.append(time.perf_counter() - start)
            total_input_tokens = int(outputs["total_input_tokens"])
        latency.append({
            "name": spec["name"],
            "mode": spec["mode"],
            "batch_size": spec["batch_size"],
            "target_len": spec["target_len"],
            "total_input_tokens": total_input_tokens,
            "latencies": latencies,
            "num_iters": spec.get("num_iters", 5),
        })

    alignment = None
    if cfg.get("alignment_num_pairs", 0) > 0:
        if cfg.get("alignment_queries") and cfg.get("alignment_docs"):
            queries = list(cfg["alignment_queries"])
            docs = list(cfg["alignment_docs"])
        else:
            queries = _make_texts(
                backend.tokenizer,
                cfg["alignment_num_pairs"],
                cfg["alignment_query_len"],
                cfg["seed"] + 500,
            )
            docs = _make_texts(
                backend.tokenizer,
                cfg["alignment_num_pairs"],
                cfg["alignment_doc_len"],
                cfg["seed"] + 700,
            )
        query_outputs = backend.encode(
            queries,
            mode="query",
            batch_size=min(batch_size, len(queries)),
            max_length=cfg["alignment_query_len"],
        )
        doc_outputs = backend.encode(
            docs,
            mode="doc",
            batch_size=min(batch_size, len(docs)),
            max_length=cfg["alignment_doc_len"],
        )
        scores = backend.score_pairs(
            queries,
            docs,
            batch_size=min(batch_size, len(queries)),
            query_max_length=cfg["alignment_query_len"],
            doc_max_length=cfg["alignment_doc_len"],
        )
        alignment = {
            "num_pairs": len(queries),
            "query_len": cfg["alignment_query_len"],
            "doc_len": cfg["alignment_doc_len"],
            "outputs": {
                "query_vecs": _sanitize_vecs(query_outputs["vecs"]),
                "doc_vecs": _sanitize_vecs(doc_outputs["vecs"]),
                "scores": [float(x) for x in scores["scores"]],
            },
        }

    return {
        "throughput": throughput,
        "latency": latency,
        "alignment": alignment,
    }
'''


KB_COLBERT_WORKER = _COLBERT_WORKER_SHARED + r'''
import importlib.util
import os


def _load_local_package(repo_dir):
    init_path = os.path.join(repo_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "kb_nano_local",
        init_path,
        submodule_search_locations=[repo_dir],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load local package from {repo_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kb_nano_local"] = module
    spec.loader.exec_module(module)
    return "kb_nano_local"


def _build_skiplist(tokenizer):
    token_ids = set()
    for symbol in string.punctuation:
        encoded = tokenizer.encode(symbol, add_special_tokens=False)
        if encoded:
            token_ids.add(int(encoded[0]))
    return token_ids


class KBColBERTBackend:
    def __init__(self, cfg):
        from transformers import AutoTokenizer

        pkg_name = _load_local_package(cfg["repo_dir"])
        embedder_mod = __import__(
            f"{pkg_name}.infra.embedder_loader",
            fromlist=["load_colbertv2_model"],
        )
        self.model_name = cfg["model"]
        self.device = cfg.get("device", "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model, self.config = embedder_mod.load_colbertv2_model(
            self.model_name,
            device=torch.device(self.device),
            dtype=torch.float16 if cfg.get("use_fp16", False) and self.device.startswith("cuda") else torch.float32,
        )
        self.model.set_skiplist(_build_skiplist(self.tokenizer))

    def _to_device(self, tensors):
        return {k: v.to(self.device) for k, v in tensors.items()}

    def encode(self, texts, mode, batch_size, max_length):
        vecs = []
        total_input_tokens = 0

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            if mode == "query":
                input_ids, attention_mask = _tensorize_queries(
                    self.tokenizer,
                    batch_texts,
                    max_length=max_length,
                    query_marker_token_id=self.config.query_marker_token_id,
                    mask_token_id=self.config.mask_token_id,
                )
            else:
                input_ids, attention_mask = _tensorize_docs(
                    self.tokenizer,
                    batch_texts,
                    max_length=max_length,
                    doc_marker_token_id=self.config.doc_marker_token_id,
                )
            total_input_tokens += int(input_ids.numel())
            inputs = self._to_device({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
            with torch.inference_mode():
                if mode == "query":
                    query_vecs = self.model.query(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                    )
                    vecs.extend(query_vecs.float().cpu().numpy())
                else:
                    doc_vecs, doc_mask = self.model.doc(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        return_mask=True,
                    )
                    doc_cpu = doc_vecs.float().cpu().numpy()
                    mask_cpu = doc_mask.cpu().numpy()
                    for item, mask in zip(doc_cpu, mask_cpu):
                        vecs.append(item[mask])

        return {
            "vecs": vecs,
            "total_input_tokens": total_input_tokens,
        }

    def score_pairs(self, queries, docs, batch_size, query_max_length, doc_max_length):
        scores = []
        for start in range(0, len(queries), batch_size):
            batch_queries = queries[start:start + batch_size]
            batch_docs = docs[start:start + batch_size]
            query_ids, query_mask = _tensorize_queries(
                self.tokenizer,
                batch_queries,
                max_length=query_max_length,
                query_marker_token_id=self.config.query_marker_token_id,
                mask_token_id=self.config.mask_token_id,
            )
            doc_ids, doc_attention_mask = _tensorize_docs(
                self.tokenizer,
                batch_docs,
                max_length=doc_max_length,
                doc_marker_token_id=self.config.doc_marker_token_id,
            )
            query_inputs = self._to_device({
                "input_ids": query_ids,
                "attention_mask": query_mask,
            })
            doc_inputs = self._to_device({
                "input_ids": doc_ids,
                "attention_mask": doc_attention_mask,
            })
            with torch.inference_mode():
                query_vecs = self.model.query(
                    query_inputs["input_ids"],
                    query_inputs["attention_mask"],
                )
                doc_vecs, doc_mask = self.model.doc(
                    doc_inputs["input_ids"],
                    doc_inputs["attention_mask"],
                    return_mask=True,
                )
                batch_scores = self.model.score(query_vecs, doc_vecs, doc_mask)
                scores.extend(batch_scores.float().cpu().numpy().tolist())
        return {"scores": scores}


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    backend = KBColBERTBackend(cfg)
    results = _run_colbert_suite(backend, cfg)
    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
'''


OFFICIAL_COLBERT_WORKER = _COLBERT_WORKER_SHARED + r'''
import types


def _build_skiplist(tokenizer):
    token_ids = set()
    for symbol in string.punctuation:
        encoded = tokenizer.encode(symbol, add_special_tokens=False)
        if encoded:
            token_ids.add(int(encoded[0]))
    return token_ids


class OfficialColBERTBackend:
    def __init__(self, cfg):
        from colbert.modeling.hf_colbert import class_factory

        self.model_name = cfg["model"]
        self.device = cfg.get("device", "cpu")
        HF_ColBERT = class_factory(self.model_name)
        self.tokenizer = HF_ColBERT.raw_tokenizer_from_pretrained(self.model_name)
        self.model = HF_ColBERT.from_pretrained(
            self.model_name,
            colbert_config=types.SimpleNamespace(dim=cfg.get("colbert_dim", 128)),
        )
        self.model = self.model.to(self.device)
        if cfg.get("use_fp16", False) and self.device.startswith("cuda"):
            self.model = self.model.to(dtype=torch.float16)
        self.model.eval()
        self.skiplist = _build_skiplist(self.tokenizer)
        self.pad_token = self.tokenizer.pad_token_id

    def _to_device(self, tensors):
        return {k: v.to(self.device) for k, v in tensors.items()}

    def _query_mask(self, input_ids):
        return input_ids.ne(self.pad_token)

    def _doc_mask(self, input_ids):
        mask = input_ids.ne(self.pad_token)
        for token_id in self.skiplist:
            mask &= input_ids.ne(token_id)
        return mask

    def encode(self, texts, mode, batch_size, max_length):
        vecs = []
        total_input_tokens = 0

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            if mode == "query":
                input_ids, attention_mask = _tensorize_queries(
                    self.tokenizer,
                    batch_texts,
                    max_length=max_length,
                    query_marker_token_id=1,
                    mask_token_id=self.tokenizer.mask_token_id,
                )
            else:
                input_ids, attention_mask = _tensorize_docs(
                    self.tokenizer,
                    batch_texts,
                    max_length=max_length,
                    doc_marker_token_id=2,
                )
            total_input_tokens += int(input_ids.numel())
            inputs = self._to_device({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
            with torch.inference_mode():
                hidden = self.model.LM(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )[0]
                projected = self.model.linear(hidden)
                if mode == "query":
                    query_mask = self._query_mask(inputs["input_ids"]).unsqueeze(-1).float()
                    query_vecs = F.normalize(projected * query_mask, p=2, dim=2)
                    vecs.extend(query_vecs.float().cpu().numpy())
                else:
                    doc_mask = self._doc_mask(inputs["input_ids"])
                    doc_vecs = F.normalize(projected * doc_mask.unsqueeze(-1).float(), p=2, dim=2)
                    doc_cpu = doc_vecs.float().cpu().numpy()
                    mask_cpu = doc_mask.cpu().numpy()
                    for item, mask in zip(doc_cpu, mask_cpu):
                        vecs.append(item[mask])

        return {
            "vecs": vecs,
            "total_input_tokens": total_input_tokens,
        }

    def score_pairs(self, queries, docs, batch_size, query_max_length, doc_max_length):
        scores = []
        for start in range(0, len(queries), batch_size):
            batch_queries = queries[start:start + batch_size]
            batch_docs = docs[start:start + batch_size]
            query_ids, query_mask = _tensorize_queries(
                self.tokenizer,
                batch_queries,
                max_length=query_max_length,
                query_marker_token_id=1,
                mask_token_id=self.tokenizer.mask_token_id,
            )
            doc_ids, doc_attention_mask = _tensorize_docs(
                self.tokenizer,
                batch_docs,
                max_length=doc_max_length,
                doc_marker_token_id=2,
            )
            query_inputs = self._to_device({
                "input_ids": query_ids,
                "attention_mask": query_mask,
            })
            doc_inputs = self._to_device({
                "input_ids": doc_ids,
                "attention_mask": doc_attention_mask,
            })
            with torch.inference_mode():
                query_hidden = self.model.LM(
                    query_inputs["input_ids"],
                    attention_mask=query_inputs["attention_mask"],
                )[0]
                doc_hidden = self.model.LM(
                    doc_inputs["input_ids"],
                    attention_mask=doc_inputs["attention_mask"],
                )[0]
                query_vecs = self.model.linear(query_hidden)
                doc_vecs = self.model.linear(doc_hidden)
                query_mask_tensor = self._query_mask(query_inputs["input_ids"]).unsqueeze(-1).float()
                doc_mask = self._doc_mask(doc_inputs["input_ids"])
                query_vecs = F.normalize(query_vecs * query_mask_tensor, p=2, dim=2)
                doc_vecs = F.normalize(doc_vecs * doc_mask.unsqueeze(-1).float(), p=2, dim=2)
                sim = doc_vecs @ query_vecs.to(dtype=doc_vecs.dtype).permute(0, 2, 1)
                sim = sim.masked_fill((~doc_mask).unsqueeze(-1), -9999)
                batch_scores = sim.max(1).values.sum(-1)
                scores.extend(batch_scores.float().cpu().numpy().tolist())
        return {"scores": scores}


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    backend = OfficialColBERTBackend(cfg)
    results = _run_colbert_suite(backend, cfg)
    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
'''


def _run_bge_benchmark(args, gpu: str, device: str, lengths: list[int], latency_batch_sizes: list[int]) -> None:
    workload = _prepare_bge_dataset_workload(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_cache_dir=args.dataset_cache_dir,
        lengths=lengths,
        latency_batch_sizes=latency_batch_sizes,
        args=args,
    )
    scenarios = workload["scenarios"]
    latency_scenarios = workload["latency_scenarios"]

    print("=" * 88)
    print("  Embedding Benchmark: kb-nano vs FlagEmbedding")
    print("=" * 88)
    print(f"  Model               : {args.model}")
    print(f"  Dataset             : {workload['dataset_name']} ({workload['num_queries']} queries, {workload['num_docs']} docs)")
    print(f"  SOTA baseline       : FlagEmbedding / BGEM3FlagModel")
    print(f"  Device              : {device}")
    print(f"  FP16                : {args.use_fp16}")
    print(f"  Return sparse       : {not args.dense_only}")
    print(f"  Return colbert      : {args.return_colbert_vecs}")
    print(f"  Throughput scenarios: {[s['name'] for s in scenarios] if scenarios else '(skipped)'}")
    print(f"  Stress lengths      : {lengths if lengths else '(none)'}")
    print(f"  Native doc max len  : {workload['native_doc_max_len']}")
    print(f"  Num texts/scenario  : {args.num_texts}")
    print(f"  Batch size          : {args.batch_size}")
    print(f"  Latency scenarios   : {[s['name'] for s in latency_scenarios] if latency_scenarios else '(skipped)'}")
    print(f"  Alignment           : {'skipped' if args.skip_alignment else f'{args.alignment_texts} texts @ len {args.alignment_len}'}")
    print(f"  Output dir          : {args.output_dir}")
    print("=" * 88)

    local_config = {
        "model": args.model,
        "seed": args.seed,
        "device": device,
        "use_fp16": args.use_fp16,
        "normalize_embeddings": True,
        "return_dense": True,
        "return_sparse": not args.dense_only,
        "return_colbert_vecs": args.return_colbert_vecs,
        "batch_size": args.batch_size,
        "warm_texts": workload["warm_texts"],
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "alignment_num_texts": 0 if args.skip_alignment else args.alignment_texts,
        "alignment_target_len": args.alignment_len,
        "alignment_texts": workload["alignment_texts"],
        "repo_dir": str(_PACKAGE_DIR),
    }

    ref_config = dict(local_config)

    ref_raw = None
    if not args.skip_reference:
        ref_raw = run_worker(
            FLAGEMBEDDING_WORKER,
            ref_config,
            f"FlagEmbedding [{args.model}]",
        )
        if ref_raw is None:
            print("ERROR: FlagEmbedding baseline worker failed.")
            sys.exit(1)

    local_raw = run_worker(
        KB_BGEM3_WORKER,
        local_config,
        f"kb-nano [{args.model}]",
    )
    if local_raw is None:
        print("ERROR: Local backend worker failed.")
        sys.exit(1)

    throughput_results = []
    if scenarios:
        print(f"\n{'=' * 104}")
        print("  THROUGHPUT SUMMARY")
        print(f"{'=' * 104}")
        print(
            f"  {'SCENARIO':<18} {'TARGET':>8} {'KB docs/s':>14} {'FLAG docs/s':>14} "
            f"{'KB tok/s':>14} {'FLAG tok/s':>14} {'SPEEDUP':>9}",
        )
        print(f"  {'-' * 96}")

        ref_tp = ref_raw["throughput"] if ref_raw else []
        for idx, local_item in enumerate(local_raw["throughput"]):
            local_docs_s = local_item["num_texts"] / local_item["elapsed"]
            local_tok_s = local_item["total_input_tokens"] / local_item["elapsed"]
            target_label = local_item.get("target_label") or str(local_item["target_len"])
            row = {
                "scenario": local_item["name"],
                "target_label": target_label,
                "target_len": local_item["target_len"],
                "num_texts": local_item["num_texts"],
                "local_elapsed": local_item["elapsed"],
                "local_docs_per_s": local_docs_s,
                "local_tok_per_s": local_tok_s,
            }

            ref_docs_s_str = "N/A"
            ref_tok_s_str = "N/A"
            speedup_str = "N/A"
            if idx < len(ref_tp):
                ref_item = ref_tp[idx]
                ref_docs_s = ref_item["num_texts"] / ref_item["elapsed"]
                ref_tok_s = ref_item["total_input_tokens"] / ref_item["elapsed"]
                speedup = _safe_div(local_tok_s, ref_tok_s)
                row["reference_elapsed"] = ref_item["elapsed"]
                row["reference_docs_per_s"] = ref_docs_s
                row["reference_tok_per_s"] = ref_tok_s
                row["speedup"] = speedup
                ref_docs_s_str = f"{ref_docs_s:,.2f}"
                ref_tok_s_str = f"{ref_tok_s:,.0f}"
                speedup_str = f"{speedup:.2f}x"

            print(
                f"  {local_item['name']:<18} {target_label:>8} "
                f"{local_docs_s:>14.2f} {ref_docs_s_str:>14} "
                f"{local_tok_s:>14,.0f} {ref_tok_s_str:>14} {speedup_str:>9}",
            )
            throughput_results.append(row)
        print(f"{'=' * 104}")

    latency_results = []
    if latency_scenarios:
        print(f"\n{'=' * 108}")
        print("  LATENCY SUMMARY")
        print(f"{'=' * 108}")
        print(
            f"  {'SCENARIO':<18} {'BS':>4} {'LEN':>6} {'KB med':>12} {'FLAG med':>12} "
            f"{'KB ms/tok':>14} {'FLAG ms/tok':>14} {'SPEEDUP':>9}",
        )
        print(f"  {'-' * 100}")

        ref_lat = ref_raw["latency"] if ref_raw else []
        for idx, local_item in enumerate(local_raw["latency"]):
            local_arr = np.asarray(local_item["latencies"], dtype=np.float64)
            local_med = float(np.median(local_arr))
            local_ms_tok = (local_med / local_item["total_input_tokens"]) * 1000
            row = {
                "scenario": local_item["name"],
                "batch_size": local_item["batch_size"],
                "target_len": local_item["target_len"],
                "local_median_s": local_med,
                "local_p99_s": float(np.percentile(local_arr, 99)),
                "local_ms_per_token": local_ms_tok,
            }

            ref_med_str = "N/A"
            ref_ms_tok_str = "N/A"
            speedup_str = "N/A"
            if idx < len(ref_lat):
                ref_item = ref_lat[idx]
                ref_arr = np.asarray(ref_item["latencies"], dtype=np.float64)
                ref_med = float(np.median(ref_arr))
                ref_ms_tok = (ref_med / ref_item["total_input_tokens"]) * 1000
                speedup = _safe_div(ref_med, local_med)
                row["reference_median_s"] = ref_med
                row["reference_p99_s"] = float(np.percentile(ref_arr, 99))
                row["reference_ms_per_token"] = ref_ms_tok
                row["speedup"] = speedup
                ref_med_str = f"{ref_med:.4f}s"
                ref_ms_tok_str = f"{ref_ms_tok:.3f}"
                speedup_str = f"{speedup:.2f}x"

            print(
                f"  {local_item['name']:<18} {local_item['batch_size']:>4} {local_item['target_len']:>6} "
                f"{local_med:>10.4f}s {ref_med_str:>12} "
                f"{local_ms_tok:>14.3f} {ref_ms_tok_str:>14} {speedup_str:>9}",
            )
            latency_results.append(row)
        print(f"{'=' * 108}")

    alignment_summary = None
    if not args.skip_alignment and ref_raw and local_raw:
        alignment_summary = _compute_alignment(
            local_raw["alignment"]["outputs"],
            ref_raw["alignment"]["outputs"],
        )
        print(f"\n{'=' * 88}")
        print("  ALIGNMENT SUMMARY")
        print(f"{'=' * 88}")
        dense = alignment_summary.get("dense")
        sparse = alignment_summary.get("sparse")
        if dense:
            print(
                f"  Dense  : avg cosine={dense.get('avg_cosine_similarity', 0.0):.8f}, "
                f"min cosine={dense.get('min_cosine_similarity', 0.0):.8f}, "
                f"avg mean abs diff={dense.get('avg_mean_abs_diff', 0.0):.8e}, "
                f"max abs diff={dense.get('max_abs_diff', 0.0):.8e}",
            )
        if sparse:
            print(
                f"  Sparse : avg key jaccard={sparse['avg_key_jaccard']:.6f}, "
                f"avg mean abs diff={sparse['avg_mean_abs_diff']:.8e}, "
                f"max abs diff={sparse['max_abs_diff']:.8e}",
            )
        print(f"{'=' * 88}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    combined = {
        "gpu": gpu,
        "model": args.model,
        "device": device,
        "local_backend": "kb-nano BGEM3ModelForInference",
        "reference_backend": "FlagEmbedding / BGEM3FlagModel" if ref_raw else None,
        "dataset": workload["dataset_name"],
        "dataset_num_queries": workload["num_queries"],
        "dataset_num_docs": workload["num_docs"],
        "dataset_num_pairs": workload["num_pairs"],
        "use_fp16": args.use_fp16,
        "return_sparse": not args.dense_only,
        "return_colbert_vecs": args.return_colbert_vecs,
        "num_texts": args.num_texts,
        "batch_size": args.batch_size,
        "lengths": lengths,
        "throughput_scenarios": throughput_results,
        "latency_scenarios": latency_results,
        "alignment": alignment_summary,
    }
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def _run_colbert_benchmark(args, gpu: str, device: str, lengths: list[int], latency_batch_sizes: list[int]) -> None:
    workload = _prepare_colbert_dataset_workload(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_cache_dir=args.dataset_cache_dir,
        lengths=lengths,
        latency_batch_sizes=latency_batch_sizes,
        args=args,
    )
    scenarios = workload["scenarios"]
    latency_scenarios = workload["latency_scenarios"]

    print("=" * 88)
    print("  Embedding Benchmark: kb-nano vs ColBERT")
    print("=" * 88)
    print(f"  Model               : {args.model}")
    print(f"  Dataset             : {workload['dataset_name']} ({workload['num_queries']} queries, {workload['num_docs']} docs)")
    print(f"  SOTA baseline       : Official ColBERT / HF_ColBERT")
    print(f"  Device              : {device}")
    print(f"  FP16                : {args.use_fp16}")
    print(f"  Query len           : {args.query_len}")
    print(f"  Throughput scenarios: {[s['name'] for s in scenarios] if scenarios else '(skipped)'}")
    print(f"  Stress doc lengths  : {lengths if lengths else '(none)'}")
    print(f"  Native doc max len  : {workload['native_doc_max_len']}")
    print(f"  Num texts/scenario  : {args.num_texts}")
    print(f"  Batch size          : {args.batch_size}")
    print(f"  Latency scenarios   : {[s['name'] for s in latency_scenarios] if latency_scenarios else '(skipped)'}")
    print(
        f"  Alignment           : "
        f"{'skipped' if args.skip_alignment else f'{args.alignment_texts} pairs @ q={args.query_len}, d={args.alignment_len}'}",
    )
    print(f"  Output dir          : {args.output_dir}")
    print("=" * 88)

    local_config = {
        "model": args.model,
        "seed": args.seed,
        "device": device,
        "use_fp16": args.use_fp16,
        "batch_size": args.batch_size,
        "query_len": args.query_len,
        "doc_len": args.latency_len,
        "colbert_dim": 128,
        "warm_queries": workload["warm_queries"],
        "warm_docs": workload["warm_docs"],
        "scenarios": scenarios,
        "latency_scenarios": latency_scenarios,
        "alignment_num_pairs": 0 if args.skip_alignment else args.alignment_texts,
        "alignment_query_len": args.query_len,
        "alignment_doc_len": args.alignment_len,
        "alignment_queries": workload["alignment_queries"],
        "alignment_docs": workload["alignment_docs"],
        "repo_dir": str(_PACKAGE_DIR),
    }
    ref_config = dict(local_config)

    ref_raw = None
    if not args.skip_reference:
        ref_raw = run_worker(
            OFFICIAL_COLBERT_WORKER,
            ref_config,
            f"ColBERT [{args.model}]",
        )
        if ref_raw is None:
            print("ERROR: ColBERT baseline worker failed.")
            sys.exit(1)

    local_raw = run_worker(
        KB_COLBERT_WORKER,
        local_config,
        f"kb-nano [{args.model}]",
    )
    if local_raw is None:
        print("ERROR: Local backend worker failed.")
        sys.exit(1)

    throughput_results = []
    if scenarios:
        print(f"\n{'=' * 116}")
        print("  THROUGHPUT SUMMARY")
        print(f"{'=' * 116}")
        print(
            f"  {'SCENARIO':<20} {'MODE':<8} {'TARGET':>8} {'KB docs/s':>14} {'REF docs/s':>14} "
            f"{'KB tok/s':>14} {'REF tok/s':>14} {'SPEEDUP':>9}",
        )
        print(f"  {'-' * 108}")

        ref_tp = ref_raw["throughput"] if ref_raw else []
        for idx, local_item in enumerate(local_raw["throughput"]):
            local_docs_s = local_item["num_texts"] / local_item["elapsed"]
            local_tok_s = local_item["total_input_tokens"] / local_item["elapsed"]
            target_label = local_item.get("target_label") or str(local_item["target_len"])
            row = {
                "scenario": local_item["name"],
                "mode": local_item["mode"],
                "target_label": target_label,
                "target_len": local_item["target_len"],
                "num_texts": local_item["num_texts"],
                "local_elapsed": local_item["elapsed"],
                "local_docs_per_s": local_docs_s,
                "local_tok_per_s": local_tok_s,
            }

            ref_docs_s_str = "N/A"
            ref_tok_s_str = "N/A"
            speedup_str = "N/A"
            if idx < len(ref_tp):
                ref_item = ref_tp[idx]
                ref_docs_s = ref_item["num_texts"] / ref_item["elapsed"]
                ref_tok_s = ref_item["total_input_tokens"] / ref_item["elapsed"]
                speedup = _safe_div(local_tok_s, ref_tok_s)
                row["reference_elapsed"] = ref_item["elapsed"]
                row["reference_docs_per_s"] = ref_docs_s
                row["reference_tok_per_s"] = ref_tok_s
                row["speedup"] = speedup
                ref_docs_s_str = f"{ref_docs_s:,.2f}"
                ref_tok_s_str = f"{ref_tok_s:,.0f}"
                speedup_str = f"{speedup:.2f}x"

            print(
                f"  {local_item['name']:<20} {local_item['mode']:<8} {target_label:>8} "
                f"{local_docs_s:>14.2f} {ref_docs_s_str:>14} "
                f"{local_tok_s:>14,.0f} {ref_tok_s_str:>14} {speedup_str:>9}",
            )
            throughput_results.append(row)
        print(f"{'=' * 116}")

    latency_results = []
    if latency_scenarios:
        print(f"\n{'=' * 120}")
        print("  LATENCY SUMMARY")
        print(f"{'=' * 120}")
        print(
            f"  {'SCENARIO':<24} {'MODE':<8} {'BS':>4} {'LEN':>6} {'KB med':>12} {'REF med':>12} "
            f"{'KB ms/tok':>14} {'REF ms/tok':>14} {'SPEEDUP':>9}",
        )
        print(f"  {'-' * 112}")

        ref_lat = ref_raw["latency"] if ref_raw else []
        for idx, local_item in enumerate(local_raw["latency"]):
            local_arr = np.asarray(local_item["latencies"], dtype=np.float64)
            local_med = float(np.median(local_arr))
            local_ms_tok = (local_med / local_item["total_input_tokens"]) * 1000
            row = {
                "scenario": local_item["name"],
                "mode": local_item["mode"],
                "batch_size": local_item["batch_size"],
                "target_len": local_item["target_len"],
                "local_median_s": local_med,
                "local_p99_s": float(np.percentile(local_arr, 99)),
                "local_ms_per_token": local_ms_tok,
            }

            ref_med_str = "N/A"
            ref_ms_tok_str = "N/A"
            speedup_str = "N/A"
            if idx < len(ref_lat):
                ref_item = ref_lat[idx]
                ref_arr = np.asarray(ref_item["latencies"], dtype=np.float64)
                ref_med = float(np.median(ref_arr))
                ref_ms_tok = (ref_med / ref_item["total_input_tokens"]) * 1000
                speedup = _safe_div(ref_med, local_med)
                row["reference_median_s"] = ref_med
                row["reference_p99_s"] = float(np.percentile(ref_arr, 99))
                row["reference_ms_per_token"] = ref_ms_tok
                row["speedup"] = speedup
                ref_med_str = f"{ref_med:.4f}s"
                ref_ms_tok_str = f"{ref_ms_tok:.3f}"
                speedup_str = f"{speedup:.2f}x"

            print(
                f"  {local_item['name']:<24} {local_item['mode']:<8} {local_item['batch_size']:>4} {local_item['target_len']:>6} "
                f"{local_med:>10.4f}s {ref_med_str:>12} "
                f"{local_ms_tok:>14.3f} {ref_ms_tok_str:>14} {speedup_str:>9}",
            )
            latency_results.append(row)
        print(f"{'=' * 120}")

    alignment_summary = None
    if not args.skip_alignment and ref_raw and local_raw:
        alignment_summary = _compute_colbert_alignment(
            local_raw["alignment"]["outputs"],
            ref_raw["alignment"]["outputs"],
        )
        print(f"\n{'=' * 92}")
        print("  ALIGNMENT SUMMARY")
        print(f"{'=' * 92}")
        for label in ("query", "doc"):
            item = alignment_summary.get(label)
            if item:
                print(
                    f"  {label.capitalize():<6}: avg cosine={item.get('avg_cosine_similarity', 0.0):.8f}, "
                    f"min cosine={item.get('min_cosine_similarity', 0.0):.8f}, "
                    f"avg mean abs diff={item.get('avg_mean_abs_diff', 0.0):.8e}, "
                    f"max abs diff={item.get('max_abs_diff', 0.0):.8e}",
                )
        scores = alignment_summary.get("scores")
        if scores:
            print(
                f"  Scores : avg mean abs diff={scores.get('avg_mean_abs_diff', 0.0):.8e}, "
                f"max abs diff={scores.get('max_abs_diff', 0.0):.8e}",
            )
        print(f"{'=' * 92}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    combined = {
        "gpu": gpu,
        "model": args.model,
        "device": device,
        "local_backend": "kb-nano ColBERTv2ModelForInference",
        "reference_backend": "Official ColBERT / HF_ColBERT" if ref_raw else None,
        "dataset": workload["dataset_name"],
        "dataset_num_queries": workload["num_queries"],
        "dataset_num_docs": workload["num_docs"],
        "dataset_num_pairs": workload["num_pairs"],
        "use_fp16": args.use_fp16,
        "num_texts": args.num_texts,
        "batch_size": args.batch_size,
        "query_len": args.query_len,
        "doc_lengths": lengths,
        "throughput_scenarios": throughput_results,
        "latency_scenarios": latency_results,
        "alignment": alignment_summary,
    }
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embedding benchmark: kb-nano vs reference embedding baselines",
    )
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    parser.add_argument(
        "--lengths",
        type=str,
        default="",
        help="Optional comma-separated fixed-length stress scenarios to add on top of dataset-backed throughput.",
    )
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mteb/scifact",
        help="HuggingFace retrieval dataset used for real query/document workload.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for the HuggingFace retrieval dataset.",
    )
    parser.add_argument("--num-texts", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use fp16. Defaults to enabled on CUDA and disabled on CPU.",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Disable sparse output and benchmark dense embeddings only.",
    )
    parser.add_argument("--return-colbert-vecs", action="store_true", default=False)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--alignment-texts", type=int, default=32)
    parser.add_argument("--alignment-len", type=int, default=128)
    parser.add_argument("--latency-len", type=int, default=128)
    parser.add_argument(
        "--latency-batch-sizes",
        type=str,
        default="1,4",
        help="Comma-separated batch sizes for latency scenarios.",
    )
    parser.add_argument("--latency-iters", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for combined results JSON.",
    )
    args = parser.parse_args()

    gpu = _detect_gpu_name()
    device = "cuda:0" if gpu != "unknown" else "cpu"
    if args.use_fp16 is None:
        args.use_fp16 = device.startswith("cuda")
    lengths = _parse_int_list(args.lengths)
    latency_batch_sizes = _parse_int_list(args.latency_batch_sizes)
    family = _detect_embedding_family(args.model)

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(repo_root / "tests" / "results" / gpu / f"{short}_embedding")

    if family == "bge_m3":
        _run_bge_benchmark(args, gpu, device, lengths, latency_batch_sizes)
        return
    if family == "colbertv2":
        _run_colbert_benchmark(args, gpu, device, lengths, latency_batch_sizes)
        return

    raise ValueError(f"Unsupported embedding family: {family}")


if __name__ == "__main__":
    main()
