#!/usr/bin/env python3
"""Drill down into the per-bucket diff between kb-nano and vLLM kernel JSONs.

For each bucket, prints the per-kernel breakdown side-by-side, plus the
"only-in-X" kernels (ones present in one engine and not the other).

Usage:
    python kb_nano/tests/debug/analyze_kernel_diff.py \\
        kb_nano/tests/logs/DeepSeek-V3.2_tp8_bs128_kprof_vllm.json \\
        kb_nano/tests/logs/DeepSeek-V3.2_tp8_bs128_kprof_kb.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Improved bucketing rules: include vLLM's custom AR + extend NCCL coverage.
BUCKET_RULES_SRC: list[tuple[str, str]] = [
    ("nccl_or_ar", r"nccl|all_reduce|allreduce|reduce_scatter|all_gather|"
                   r"cross_device_reduce|sendrecv"),
    ("flashinfer_ar", r"flashinfer.*allreduce|trtllm.*allreduce|allreduce_fusion"),
    ("attention_mla", r"flash_mla|fmla_|flashmla"),
    ("attention_dense", r"flash_attn|fmha|flash_fwd"),
    ("attention_sparse_attn", r"sparse_attn_fwd"),
    ("sparse_indexer", r"fp8_mqa|fp8_paged_mqa|mqa_logits|topk_per_row|"
                       r"sparse_attn_indexer|paged_mqa_logits_metadata"),
    ("gemm_fp8_deepgemm", r"deep_gemm|fp8_gemm_nt|fp8_blockwise|gemm_grouped|"
                          r"gemm_fp8|m_grouped_gemm"),
    ("gemm_bf16_cublas", r"cutlass|cublas|gemm.*bf16|gemm.*fp16|sm90.*gemm|"
                         r"hgemm|nvjet|splitkreduce"),
    ("moe_dispatch", r"fused_moe|fused_experts|moe_align|moe_sum|grouped_topk|"
                     r"grouped_softmax|count_and_sort"),
    ("moe_router", r"router|topk|noaux"),
    ("norm_quant_fused", r"fused.*norm.*quant|fused.*rms.*quant|norm_quant|"
                         r"rms_quant"),
    ("rmsnorm", r"rms_norm|rmsnorm|fused_add_rms"),
    ("quant_per_token", r"per_token_group_quant|per_token_quant|act_quant|"
                        r"fp8_quant"),
    ("act_silu", r"silu|gelu|swiglu"),
    ("sampling", r"sample|argmax|multinomial|softmax"),
    ("copy_memcpy", r"memcpy|memset|copy_kernel|d2h|h2d"),
    ("triton_fused", r"triton_fused|triton_per_fused|triton_red"),
    ("elementwise", r"elementwise|vectorized|pointwise|index_kernel|gather|"
                    r"scatter|catarray"),
]

RULES = [(b, re.compile(p)) for b, p in BUCKET_RULES_SRC]


def bucket_for(name: str) -> str:
    n = name.lower()
    for b, pat in RULES:
        if pat.search(n):
            return b
    return "other"


def load(path: Path) -> dict[str, dict]:
    """Return {kernel_name: {count, self_cuda_us}}."""
    data = json.loads(path.read_text())
    by = {}
    for k in data["by_kernel"]:
        by[k["name"]] = {"count": k["count"], "us": k["self_cuda_us"]}
    return by


def diff_buckets(v: dict, k: dict) -> dict:
    """Return per-bucket aggregated diff."""
    bv: dict[str, dict] = {}
    bk: dict[str, dict] = {}
    for src, dst in [(v, bv), (k, bk)]:
        for name, vals in src.items():
            b = bucket_for(name)
            d = dst.setdefault(b, {"count": 0, "us": 0.0})
            d["count"] += vals["count"]
            d["us"] += vals["us"]
    out = {}
    for b in set(bv) | set(bk):
        v_us = bv.get(b, {}).get("us", 0.0)
        k_us = bk.get(b, {}).get("us", 0.0)
        v_ct = bv.get(b, {}).get("count", 0)
        k_ct = bk.get(b, {}).get("count", 0)
        out[b] = {"v_us": v_us, "k_us": k_us, "v_ct": v_ct, "k_ct": k_ct,
                  "delta_us": k_us - v_us}
    return out


def print_buckets(diff: dict) -> None:
    rows = sorted(diff.items(), key=lambda kv: abs(kv[1]["delta_us"]),
                  reverse=True)
    total_v = sum(d["v_us"] for d in diff.values())
    total_k = sum(d["k_us"] for d in diff.values())
    print(f"\nTOTAL  vLLM={total_v/1000:.2f} ms   kb={total_k/1000:.2f} ms"
          f"   delta={(total_k-total_v)/1000:+.2f} ms")
    print(f"\n  {'bucket':<24} {'vLLM ms':>9} {'kb ms':>9} {'delta':>9}"
          f"  {'kb/vLLM':>8}  {'vLLM ct':>8} {'kb ct':>8}")
    print(f"  {'-'*24} {'-'*9} {'-'*9} {'-'*9}  {'-'*8}  {'-'*8} {'-'*8}")
    for b, d in rows:
        ratio = (d["k_us"] / d["v_us"]) if d["v_us"] > 0 else float("inf")
        ratio_s = f"{ratio:>7.2f}x" if d["v_us"] > 0 else "    inf"
        print(f"  {b:<24} {d['v_us']/1000:>9.2f} {d['k_us']/1000:>9.2f}"
              f" {d['delta_us']/1000:>+9.2f}  {ratio_s}"
              f"  {d['v_ct']:>8d} {d['k_ct']:>8d}")


def print_bucket_kernels(bucket: str, v: dict, k: dict, top: int = 25) -> None:
    """Print per-kernel breakdown for kernels in a given bucket."""
    print(f"\n{'='*80}")
    print(f"BUCKET: {bucket}")
    print(f"{'='*80}")

    v_in = {n: vals for n, vals in v.items() if bucket_for(n) == bucket}
    k_in = {n: vals for n, vals in k.items() if bucket_for(n) == bucket}

    all_names = set(v_in) | set(k_in)
    rows = []
    for n in all_names:
        vu = v_in.get(n, {}).get("us", 0.0)
        ku = k_in.get(n, {}).get("us", 0.0)
        vc = v_in.get(n, {}).get("count", 0)
        kc = k_in.get(n, {}).get("count", 0)
        rows.append((n, vu, ku, ku - vu, vc, kc))
    rows.sort(key=lambda r: max(r[1], r[2]), reverse=True)

    print(f"\n  {'vLLM ms':>9} {'kb ms':>9} {'delta':>9}  {'vLLM ct':>8} {'kb ct':>8}  name")
    print(f"  {'-'*9} {'-'*9} {'-'*9}  {'-'*8} {'-'*8}  {'-'*60}")
    for n, vu, ku, delta, vc, kc in rows[:top]:
        marker = ""
        if vu == 0 and ku > 0:
            marker = " *kb-only*"
        elif ku == 0 and vu > 0:
            marker = " *vLLM-only*"
        short = n[:90].replace("\n", " ")
        print(f"  {vu/1000:>9.2f} {ku/1000:>9.2f} {delta/1000:>+9.2f}"
              f"  {vc:>8d} {kc:>8d}  {short}{marker}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("vllm_json", type=Path)
    p.add_argument("kb_json", type=Path)
    p.add_argument("--bucket", action="append", default=None,
                   help="Bucket(s) to drill into. Repeat. Default: top-5 by abs delta.")
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    v = load(args.vllm_json)
    k = load(args.kb_json)
    print(f"vLLM kernels: {len(v)}, kb-nano kernels: {len(k)}")
    diff = diff_buckets(v, k)
    print_buckets(diff)

    if args.bucket:
        buckets = args.bucket
    else:
        buckets = [b for b, _ in sorted(diff.items(),
                                         key=lambda kv: abs(kv[1]["delta_us"]),
                                         reverse=True)[:6]]
        print(f"\nDrilling into top {len(buckets)} buckets by |delta|: {buckets}")

    for b in buckets:
        print_bucket_kernels(b, v, k, args.top)


if __name__ == "__main__":
    main()
