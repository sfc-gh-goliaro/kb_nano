#!/usr/bin/env python3
"""Build real-prompt benchmark datasets for kb-nano LLM throughput tests.

Three scenarios are produced, each as a JSON artefact under
``kb_nano/bench/datasets/real_prompts/``:

  * ``prefill-heavy``  -- LongBench (16 tasks)        n=400,  prompt cap=4096, decode cap=256
  * ``balanced``       -- WildChat-1M (EN, clean)     n=1000, prompt band [256, 1024], decode band [256, 1024]
  * ``decode-heavy``   -- OpenThoughts-114k           n=500,  prompt cap=1024, decode cap=2048

Each request stores the **raw chat text** (a list of ``{role, content}``
messages plus the assistant response text) so the benchmark runner can
re-tokenize the same workload for any target model. The reference
Llama-3.1-8B-Instruct tokenizer is used at build time only for:

  * Filtering: prefer prompts whose chat-templated length already fits
    under the per-scenario prompt cap / band; otherwise mark the prompt
    as oversized so the runner left-truncates it.
  * Summary statistics in ``meta.json`` (token-length distributions).

Prompt cap policy (LongBench, OpenThoughts):
  1. Prefer prompts whose Llama chat-templated token length is <= cap.
  2. If not enough, supplement with longer prompts (``oversized_at_build=True``);
     the runtime is expected to left-truncate them per its own tokenizer.

Decode cap policy:
  The cap is recorded in ``meta.config.decode_cap``. The build script
  reports the **capped** decode length distribution in ``stats``, but the
  per-request artefact stores the full assistant response text so any
  tokenizer can compute its own ``min(len, decode_cap)``.

Usage:
    python -m kb_nano.bench.datasets.build_real_prompts
    python -m kb_nano.bench.datasets.build_real_prompts --scenario prefill-heavy
    python -m kb_nano.bench.datasets.build_real_prompts --output-dir /tmp/rp
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from kb_nano import KB_ROOT


TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 0
DATA_DIR = KB_ROOT / "bench" / "datasets" / "real_prompts"

# 16 LongBench tasks spanning QA, summarization, code, and few-shot retrieval.
LB_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]

SCENARIOS: dict[str, dict] = {
    "prefill-heavy": {
        "source": "longbench",
        "dataset": "Xnhyacinth/LongBench",
        "n_requests": 400,
        "prompt_cap": 4096,
        "decode_cap": 256,
    },
    "balanced": {
        "source": "wildchat",
        "dataset": "allenai/WildChat-1M",
        "n_requests": 1000,
        # Symmetric bands keep mean(prompt) ~= mean(decode); without a
        # decode floor WildChat's short first-turn replies dominated and
        # pushed the prompt/decode ratio to ~2:1.
        "prompt_band": (256, 1024),
        "decode_cap": 1024,
        "decode_floor": 256,
    },
    "decode-heavy": {
        "source": "openthoughts",
        "dataset": "open-thoughts/OpenThoughts-114k",
        "n_requests": 500,
        "prompt_cap": 1024,
        "decode_cap": 2048,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summarize(arr) -> dict:
    arr = np.asarray(arr, dtype=np.int64)
    return {
        "n": int(arr.size),
        "min": int(arr.min()) if arr.size else 0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "median": int(np.median(arr)) if arr.size else 0,
        "p95": int(np.percentile(arr, 95)) if arr.size else 0,
        "p99": int(np.percentile(arr, 99)) if arr.size else 0,
        "max": int(arr.max()) if arr.size else 0,
        "total": int(arr.sum()),
    }


def _select_with_truncation(short_pool, long_pool, n_requests, seed):
    """Pick n_requests records, preferring uncapped ones; oversized as fallback.

    Each pool entry is a 6-tuple ``(messages, assistant_text,
    prompt_token_count, capped_decode_count, source_id, oversized)``. Returns
    the chosen list and the count of oversized records that were used.
    """
    rng = random.Random(seed)
    rng.shuffle(short_pool)
    rng.shuffle(long_pool)
    if len(short_pool) >= n_requests:
        return short_pool[:n_requests], 0
    chosen = list(short_pool)
    needed = n_requests - len(chosen)
    chosen.extend(long_pool[:needed])
    n_oversized = min(needed, len(long_pool))
    if len(chosen) < n_requests:
        print(
            f"  WARNING: only {len(chosen)} / {n_requests} requests "
            f"available (short={len(short_pool)} oversized={len(long_pool)})"
        )
    return chosen, n_oversized


# ---------------------------------------------------------------------------
# LongBench (prefill-heavy)
# ---------------------------------------------------------------------------
def build_longbench(tok, n_requests, prompt_cap, decode_cap, seed=SEED):
    print(f"\n[longbench] target n={n_requests} prompt_cap={prompt_cap} "
          f"decode_cap={decode_cap}")
    records = []
    for t in tqdm(LB_TASKS, desc="LB load"):
        try:
            ds = load_dataset("Xnhyacinth/LongBench", t, split="test")
        except Exception as e:
            print(f"  skipping task {t}: {e}")
            continue
        for ex in ds:
            ctx = ex.get("context", "") or ""
            inp = ex.get("input", "") or ""
            answers = ex.get("answers") or []
            if isinstance(answers, list):
                answer = str(answers[0]) if answers else ""
            else:
                answer = str(answers)
            if not (ctx or inp):
                continue
            text = (ctx + "\n\n" + inp).strip() if ctx else inp
            records.append({
                "task": t,
                "id": ex.get("_id") or ex.get("id") or "",
                "text": text,
                "answer": answer,
            })
    print(f"  collected {len(records):,} examples; tokenizing...")

    short_pool = []
    long_pool = []
    for r in tqdm(records, desc="LB tok"):
        messages = [{"role": "user", "content": r["text"]}]
        try:
            full_ids = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
            ans_ids = tok.encode(r["answer"] or " ",
                                 add_special_tokens=False)
        except Exception:
            continue
        prompt_count = len(full_ids)
        capped_decode = max(1, min(len(ans_ids), decode_cap))
        sid = f"{r['task']}::{r['id']}"
        oversized = prompt_count > prompt_cap
        entry = (
            messages,
            r["answer"] or " ",
            prompt_count,
            capped_decode,
            sid,
            oversized,
        )
        if oversized:
            long_pool.append(entry)
        else:
            short_pool.append(entry)

    print(f"  pool: {len(short_pool):,} <= {prompt_cap} tok, "
          f"{len(long_pool):,} oversized candidates")
    chosen, n_over = _select_with_truncation(
        short_pool, long_pool, n_requests, seed)
    print(f"  selected {len(chosen)} prompts ({n_over} oversized)")
    return chosen, n_over


# ---------------------------------------------------------------------------
# WildChat-1M (balanced)
# ---------------------------------------------------------------------------
def build_wildchat(tok, n_requests, prompt_band, decode_cap,
                   decode_floor=0, seed=SEED):
    lo, hi = prompt_band
    print(f"\n[wildchat] target n={n_requests} prompt_band=[{lo}, {hi}] "
          f"decode_band=[{decode_floor}, {decode_cap}]")
    ds = load_dataset(
        "allenai/WildChat-1M", split="train", streaming=True,
    ).shuffle(seed=seed, buffer_size=10_000)

    target = n_requests * 5  # gather extras for stable selection
    candidates = []
    n_processed = 0
    n_skip_decode = 0
    pbar = tqdm(total=target, desc="WC stream")
    for item in ds:
        if len(candidates) >= target:
            break
        n_processed += 1
        if item.get("toxic") or item.get("redacted"):
            continue
        if item.get("language") != "English":
            continue
        conv = item.get("conversation", []) or []

        history = []
        emitted = False
        for turn_idx, t in enumerate(conv):
            role = t.get("role")
            content = t.get("content", "")
            if not isinstance(content, str):
                continue
            if role == "user":
                history.append({"role": "user", "content": content})
                if not emitted:
                    try:
                        prompt_ids = tok.apply_chat_template(
                            history,
                            tokenize=True, add_generation_prompt=True,
                        )
                    except Exception:
                        prompt_ids = None
                    if prompt_ids is not None and lo <= len(prompt_ids) <= hi:
                        nxt = (conv[turn_idx + 1]
                               if turn_idx + 1 < len(conv) else None)
                        if (nxt and nxt.get("role") == "assistant"
                                and isinstance(nxt.get("content"), str)):
                            ans_text = nxt["content"]
                            ans_ids = tok.encode(
                                ans_text, add_special_tokens=False)
                            capped_decode = max(
                                1, min(len(ans_ids), decode_cap))
                            if capped_decode < decode_floor:
                                n_skip_decode += 1
                                continue
                            sid = (
                                f"{item.get('conversation_hash', '?')}"
                                f"::turn{turn_idx}"
                            )
                            candidates.append((
                                # Snapshot of the chat history at emit time.
                                [dict(m) for m in history],
                                ans_text,
                                len(prompt_ids),
                                capped_decode,
                                sid,
                                False,
                            ))
                            pbar.update(1)
                            emitted = True
            elif role == "assistant":
                history.append({"role": "assistant", "content": content})
        if n_processed % 500 == 0:
            pbar.set_postfix({"scanned": n_processed,
                              "candidates": len(candidates),
                              "skip_dec": n_skip_decode})
    pbar.close()
    print(f"  scanned {n_processed:,} convs, collected {len(candidates):,} "
          f"candidates  (skipped {n_skip_decode:,} for decode<{decode_floor})")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = candidates[:n_requests]
    if len(chosen) < n_requests:
        print(f"  WARNING: only {len(chosen)} / {n_requests} candidates")
    print(f"  selected {len(chosen)} prompts")
    return chosen, 0


# ---------------------------------------------------------------------------
# OpenThoughts-114k (decode-heavy)
# ---------------------------------------------------------------------------
def build_openthoughts(tok, n_requests, prompt_cap, decode_cap, seed=SEED):
    print(f"\n[openthoughts] target n={n_requests} prompt_cap={prompt_cap} "
          f"decode_cap={decode_cap}")
    ot = load_dataset("open-thoughts/OpenThoughts-114k", split="train")

    records = []
    for ex in tqdm(ot, desc="OT collect"):
        sys_p = ex.get("system", "") or ""
        convo = ex.get("conversations", []) or []
        user, asst = "", ""
        for t in convo:
            role = t.get("from", "")
            v = t.get("value", "")
            if role in ("user", "human") and not user:
                user = v
            elif role in ("assistant", "gpt") and not asst:
                asst = v
        if user and asst:
            records.append({"system": sys_p, "user": user, "asst": asst})

    rng = random.Random(seed)
    rng.shuffle(records)

    # Tokenize until we have enough short prompts (most OT prompts are short).
    target_short = max(n_requests * 3, n_requests + 200)
    short_pool = []
    long_pool = []
    pbar = tqdm(total=target_short, desc="OT tok (short)")
    for i, r in enumerate(records):
        if len(short_pool) >= target_short:
            break
        msgs = []
        if r["system"]:
            msgs.append({"role": "system", "content": r["system"]})
        msgs.append({"role": "user", "content": r["user"]})
        try:
            prompt_ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
            )
            ans_ids = tok.encode(r["asst"], add_special_tokens=False)
        except Exception:
            continue
        prompt_count = len(prompt_ids)
        capped_decode = max(1, min(len(ans_ids), decode_cap))
        sid = f"ot::{i}"
        oversized = prompt_count > prompt_cap
        entry = (msgs, r["asst"], prompt_count, capped_decode, sid, oversized)
        if oversized:
            long_pool.append(entry)
        else:
            short_pool.append(entry)
            pbar.update(1)
    pbar.close()
    print(f"  pool: {len(short_pool):,} <= {prompt_cap} tok, "
          f"{len(long_pool):,} oversized candidates")
    chosen, n_over = _select_with_truncation(
        short_pool, long_pool, n_requests, seed)
    print(f"  selected {len(chosen)} prompts ({n_over} oversized)")
    return chosen, n_over


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------
def write_artifact(name, scenario_cfg, chosen, n_oversized, out_dir):
    # chosen entry layout:
    #   (messages, assistant_text, prompt_token_count, capped_decode_count,
    #    source_id, oversized_at_build)
    prompt_lens = [c[2] for c in chosen]
    decode_lens = [c[3] for c in chosen]
    artifact = {
        "scenario": name,
        "tokenizer": TOKENIZER,
        "seed": SEED,
        "n_requests": len(chosen),
        "config": scenario_cfg,
        "stats": {
            "prompt_tokens": _summarize(prompt_lens),
            "decode_tokens": _summarize(decode_lens),
            "n_prompts_oversized": int(n_oversized),
        },
        "requests": [
            {
                "messages": c[0],
                "assistant_text": c[1],
                "source_id": c[4],
                "oversized_at_build": bool(c[5]),
            }
            for c in chosen
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  -> wrote {out_path}  ({size_mb:.2f} MB)")
    print(f"     prompts:  {artifact['stats']['prompt_tokens']}")
    print(f"     decodes:  {artifact['stats']['decode_tokens']}")
    print(f"     totals: prompt={artifact['stats']['prompt_tokens']['total']:,}"
          f"  decode={artifact['stats']['decode_tokens']['total']:,}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def build(name, out_dir, tok=None):
    cfg = SCENARIOS[name]
    if tok is None:
        print(f"Loading tokenizer: {TOKENIZER}", flush=True)
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    src = cfg["source"]
    if src == "longbench":
        chosen, n_over = build_longbench(
            tok, cfg["n_requests"], cfg["prompt_cap"], cfg["decode_cap"])
    elif src == "wildchat":
        chosen, n_over = build_wildchat(
            tok, cfg["n_requests"], cfg["prompt_band"], cfg["decode_cap"],
            decode_floor=cfg.get("decode_floor", 0))
    elif src == "openthoughts":
        chosen, n_over = build_openthoughts(
            tok, cfg["n_requests"], cfg["prompt_cap"], cfg["decode_cap"])
    else:
        raise ValueError(f"unknown source: {src}")
    write_artifact(name, cfg, chosen, n_over, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()) + ["all"], default="all")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print(f"Output directory: {out_dir}")
    print(f"Loading tokenizer: {TOKENIZER}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for name in names:
        build(name, out_dir, tok=tok)


if __name__ == "__main__":
    main()
