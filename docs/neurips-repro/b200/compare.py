#!/usr/bin/env python3
"""Compare the B200 sweep against the paper table (`../table_results.tex`).

Maps each paper row to its result directory by substring, extracts a speedup and
the best available alignment metric (bench schemas differ per row), and flags
rows that land meaningfully below the published number.

    python compare.py [--gpu B200] [--tol 0.10]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean

# (category, row, result-dir substring, paper speedup, paper alignment)
ROWS = [
    ("Dense/MoE",  "Llama-3.1",          "Llama-3.1-8B-Instruct",      1.04, "408.5 tok"),
    ("Dense/MoE",  "DeepSeek-V3.2",      "DeepSeek-V3.2",              0.84, "294.1 tok"),
    ("Dense/MoE",  "Mixtral",            "Mixtral-8x7B",               0.97, "108.9 tok"),
    ("Dense/MoE",  "BitNet 1.58b",       "bitnet",                     1.12, "Top-20 100%"),
    # The paper row is the 20b at TP=1. table_results.tex names no size, but
    # bench/eval/config.py's MODEL_KEY_TO_DEFAULT_HF -- whose other entries match the
    # paper's rows one-for-one, down to Qwen3-VL-8B-*FP8* -- maps "gpt_oss" to
    # openai/gpt-oss-20b, and bench_vllm.py defaults --tp to 1 with --num-seqs 1000
    # ("WildChat 1K"). The 120b appears only in the out-of-repo planning spreadsheet and
    # the kernel shape registry. An earlier revision of this file mapped the row to the
    # 120b; that was my guess and it inverted the verdict for this row.
    ("Dense/MoE",  "GPT-OSS (MXFP4)",    "gpt-oss-20b",                1.02, "599.6 tok"),
    ("Dense/MoE",  "GPT-OSS 120b (extra)", "gpt-oss-120b",             1.02, "—"),
    ("Dense/MoE",  "EAGLE-3",            "EAGLE3",                     0.98, "Top-20 100%"),
    ("Dense/MoE",  "Gemma-4",            "gemma-4",                    1.00, "Top-20 ~100%"),
    ("Linear-attn", "Mamba",             "mamba-2.8b",                 1.05, "541.3 tok"),
    ("Linear-attn", "Mamba2",            "Mamba-Codestral",            0.97, "555.9 tok"),
    ("Linear-attn", "RWKV-7",            "rwkv7",                      1.18, "593.8 tok"),
    ("Linear-attn", "GLA",               "gla-2.7B",                   1.85, "645.5 tok"),
    ("Linear-attn", "RetNet",            "retnet-2.7B",                1.86, "647.0 tok"),
    ("Linear-attn", "Qwen-3-Next",       "Qwen3-Next",                 1.24, "487.4 tok"),
    ("Linear-attn", "Kimi-Linear",       "Kimi-Linear",                1.20, "Top-20 99.22%"),
    ("Linear-attn", "TTT-E2E",           "ttt",                        1.10, "NLL diff 8.3e-2"),
    ("Linear-attn", "Jamba",             "Jamba",                      1.02, "415.4 tok"),
    ("Vision/AV",  "FLUX.1-Dev",         "FLUX.1-dev",                 1.01, "img cos 0.995"),
    ("Vision/AV",  "HunyuanVideo-1.5",   "HunyuanVideo",               0.97, "cos 0.924 / 12.94dB"),
    ("Vision/AV",  "SDXL",               "stable-diffusion-xl",        1.17, "latent cos 0.982"),
    ("Vision/AV",  "SAM3.1",             "sam3",                       1.05, "0.980/0.949/0.975"),
    ("Vision/AV",  "Whisper",            "whisper",                    0.95, "388.7 / 444"),
    ("Vision/AV",  "CosyVoice3",         "CosyVoice3",                 2.13, "mel cos 0.999"),
    ("Multimodal", "Qwen2-VL",           "Qwen2-VL",                   0.91, "539.4 tok"),
    ("Multimodal", "Qwen3-VL",           "Qwen3-VL",                   1.39, "368.5 tok"),
    ("Multimodal", "Qwen-2.5-Omni",      "Qwen2.5-Omni",               2.02, "exact match 36.2%"),
    ("Multimodal", "SigLIP-2",           "siglip2",                    0.93, "cos 1.000"),
    ("Multimodal", "DINOv3",             "dinov3",                     0.99, "cos 1.000"),
    ("Multimodal", "SwinV2",             "swinv2",                     1.17, "cos 1.000"),
    ("Edge/Det",   "MobileNetV4",        "mobilenetv4",                1.15, "cos 1.000"),
    ("Edge/Det",   "ConvNeXtV2",         "convnextv2",                 0.99, "cos 1.000"),
    ("Edge/Det",   "EfficientNetV2",     "efficientnetv2",             1.05, "cos 1.000"),
    ("Edge/Det",   "YOLOv10",            "yolov10",                    1.06, "1.000 x3"),
    ("Edge/Det",   "RTDetrV2",           "rtdetr",                     1.08, "1.000 x3"),
    ("3D/Robotics", "3DGS",              "3dgs",                       0.99, "cos 1.000"),
    ("3D/Robotics", "InstantNGP",        "instantngp",                 0.97, "RGBA cos 1.000"),
    ("3D/Robotics", "PointTransformerV3", "point",                     1.00, "feat cos 0.9796"),
    ("3D/Robotics", "OpenFold3",         "openfold",                   1.03, "100% pass"),
    ("3D/Robotics", "Pi0",               "pi0",                        3.48, "cos+MSE 0.9997"),
    ("3D/Robotics", "DP3",               "dp3",                        1.42, "cos 1.000, MSE 0"),
    ("Recsys/Spec", "DLRMv2",            "dlrm",                       1.06, "cos 1.000"),
    ("Recsys/Spec", "LightGCN",          "lightgcn",                   1.03, "cos 1.000"),
    ("Recsys/Spec", "BGE-M3",            "bge-m3",                     1.06, "cos 0.999012"),
    ("Recsys/Spec", "ColBERTv2",         "colbert",                    3.08, "cos 0.999992"),
    ("Recsys/Spec", "LLaDA",             "LLaDA",                      1.07, "98.35%, 0.99983"),
    ("World model", "Oasis",             "oasis",                      1.29, "n/a"),
    ("World model", "V-JEPA 2",          "vjepa",                      1.01, "cos 1.000"),
]

RATE_KEYS = ("images_per_second", "items_per_second", "tokens_per_second",
             "tok_per_s", "samples_per_second", "frames_per_second",
             "steps_per_second", "requests_per_second", "utterances_per_second",
             "throughput_req_s", "inferences_per_second",
             "videos_per_second", "throughput")
REF_KEYS = ("vllm", "vllm_omni", "timm", "diffusers", "reference", "ref", "sota",
            "transformers", "sglang", "torchrec", "gsplat", "pyngp", "openpi",
            "fla", "pyg", "fastdllm", "open_oasis", "dp3", "ptv3", "sam3")


def walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (str(i),))
    else:
        yield path, obj


def _dicts(obj):
    """Every dict in the document, outermost first."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _dicts(v)


def rate_of(entry):
    for k in RATE_KEYS:
        v = entry.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return v
    return None


def speedups(doc):
    """Explicit speedup fields, else paired throughput lists, else rate pairs."""
    # Benches spell the same quantity three ways: "speedup",
    # bench_image_cls's ``comparisons.<ref>.throughput_ratio``, and
    # bench_recsys's ``throughput.ratio``.
    # bench_eagle3 names its field after the reference ("speedup_vs_sglang"), which no
    # generic key matched -- the row printed as "RAN, NO BASELINE" despite having a
    # perfectly good 0.99x in the file. Take the throughput scenarios only: the same doc
    # also carries latency_scenarios, whose ratio is a different quantity.
    sp = [v for p, v in walk(doc)
          if p[-1] in ("speedup", "throughput_ratio", "ratio",
                       "ratio_vs_reference")
          and isinstance(v, (int, float)) and v > 0]
    if not sp:
        sp = [v for p, v in walk(doc)
              if p[-1].startswith("speedup_vs_")
              and "latency" not in "/".join(str(x) for x in p)
              and isinstance(v, (int, float)) and v > 0]
    if sp:
        return sp
    for node in _dicts(doc):
        fk = node.get("fastkernels")
        if not (isinstance(fk, dict) and isinstance(fk.get("throughput"), list)):
            continue
        for k in REF_KEYS:
            ref = node.get(k)
            if isinstance(ref, dict) and isinstance(ref.get("throughput"), list):
                by = {e.get("name"): e for e in ref["throughput"] if isinstance(e, dict)}
                out = []
                for e in fk["throughput"]:
                    r = by.get(e.get("name")) if isinstance(e, dict) else None
                    if r:
                        a, b = rate_of(e), rate_of(r)
                        if a and b:
                            out.append(a / b)
                if out:
                    return out
    # bench_vjepa2 / bench_image_cls style: {"throughput": {"ours": {...},
    # "reference": {...}}} -- dicts of rates rather than parallel lists.
    for node in _dicts(doc):
        ours = node.get("ours")
        if not isinstance(ours, dict):
            continue
        a = rate_of(ours)
        if not a:
            continue
        for k in REF_KEYS:
            ref = node.get(k)
            if isinstance(ref, dict):
                b = rate_of(ref)
                if b:
                    return [a / b]
    tr = _timed_run_ratio(doc)
    if tr:
        return tr
    ours = [v for k, v in doc.items()
            if isinstance(v, (int, float)) and k.startswith("fastkernels_")
            and ("per_sec" in k or "per_s" in k)]
    theirs = [v for k, v in doc.items()
              if isinstance(v, (int, float))
              and (k.startswith("ref_") or k.startswith("vllm_"))
              and ("per_sec" in k or "per_s" in k)]
    if len(ours) == 1 and len(theirs) == 1 and theirs[0]:
        return [ours[0] / theirs[0]]
    return []


def _timed_run_ratio(doc):
    """bench_ttt_e2e records raw per-sequence timings, not a speedup.

    Shape: results[].per_seq[].{jax,fastkernels}.run_times_s. Speedup is
    reference-time / our-time, using medians to match the bench's own summary.
    """
    from statistics import median
    ratios = []
    for node in _dicts(doc):
        ours = node.get("fastkernels")
        if not isinstance(ours, dict) or "run_times_s" not in ours:
            continue
        for ref_key in ("jax", "reference", "ref"):
            ref = node.get(ref_key)
            if isinstance(ref, dict) and ref.get("run_times_s"):
                a, b = ref["run_times_s"], ours["run_times_s"]
                if a and b:
                    ratios.append(median(a) / median(b))
                break
    return ratios


def _scale(doc):
    """Workload size, so a reduced-scale run is visibly not a full-scale one."""
    for k in ("num_seqs", "num_prompts", "num_requests", "num_images",
              "num_videos", "num_items"):
        v = doc.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None


def alignment(doc):
    # Prefer a deterministic metric when the bench provides one. CosyVoice3's
    # end-to-end mel cosine (~0.86) carries talker sampling noise, while its
    # code2wav equivalence check (~0.9994) is the figure the paper reports;
    # averaging them together hides the reproduction.
    det = [(p, v) for p, v in walk(doc)
           if isinstance(v, (int, float)) and len(p) >= 2
           and "code2wav" in p[-2].lower() and "cosine" in p[-1].lower()]
    if det:
        return f"code2wav cos {det[0][1]:.4f}"
    cos = [v for p, v in walk(doc)
           if isinstance(v, (int, float)) and re.search(r"cos(ine)?", p[-1], re.I)
           and not re.search(r"min|max|std", p[-1], re.I)]
    if cos:
        nan = [v for v in cos if v != v]
        if nan:
            return f"cos NaN ({len(nan)}/{len(cos)}) <- reference produced nothing"
        return f"cos {mean(cos):.4f}"
    for pat in (r"matching_tokens", r"exact_match", r"match", r"pass", r"top.?20"):
        hit = [(p[-1], v) for p, v in walk(doc)
               if isinstance(v, (int, float)) and re.search(pat, p[-1], re.I)]
        if hit:
            k, v = hit[0]
            return f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="B200")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="flag rows below (1-tol) x paper speedup")
    ap.add_argument("--after", default=None, metavar="MM-DD HH:MM",
                    help="mark results older than this as STALE (i.e. from the "
                         "reduced-scale smoke pass rather than the full sweep)")
    a = ap.parse_args()
    root = Path(f"/home/yak/kb_nano/tests/results/{a.gpu}")

    # Result files are not uniformly named or placed: the embedding bench writes
    # ``embedding/<timestamp>/results.json`` for several models, bench_dllm writes
    # ``<model>/<task>_<config>.json``, and some benches key the directory off a
    # short name. Index every JSON and match rows on the path *and* the model
    # string inside, rather than assuming <pattern>/results.json.
    index = []  # (haystack, path, mtime)
    # Nearly everything lands under tests/results/<gpu>/, but a few benches use
    # their own layout (bench_openfold3 writes tests/results/openfold3/<gpu>_<dtype>/).
    # Include sibling trees that contain a <gpu>-tagged subdirectory, rather than
    # walking every GPU's results.
    scan_root = root.parent if root.parent.exists() else root
    roots = [root] if root.exists() else []
    if root.parent.exists():
        for d in root.parent.iterdir():
            if not d.is_dir() or d == root:
                continue
            try:
                if any(sub.is_dir() and a.gpu.lower() in sub.name.lower()
                       for sub in d.iterdir()):
                    roots.append(d)
            except OSError:
                pass
    for base in roots:
        for f in base.rglob("*.json"):
            # Skip bulk artifacts (per-request token dumps run to tens of MB).
            try:
                if f.stat().st_size > 20 * 1024 * 1024:
                    continue
            except OSError:
                continue
            if f.name in ("_ds_manifest.json",):
                continue
            try:
                doc = json.load(open(f))
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            names = []
            for path_keys, v in walk(doc):
                if isinstance(v, str) and path_keys and path_keys[-1] in (
                        "model", "model_key", "model_name", "timm_name",
                        "draft_model", "variant", "scenario", "sota"):
                    names.append(v)
            models = doc.get("models")
            if isinstance(models, dict):
                names.extend(models.keys())
            hay = (str(f.relative_to(scan_root)) + " " + " ".join(names)).lower()
            index.append((hay, f, doc))
    print(f"{'Category':<12} {'Row':<20} {'paper':>6} {a.gpu:>7} {'ratio':>7}  "
          f"{'measured':<12}  alignment")
    print("-" * 104)
    behind, missing = [], []
    for cat, row, pat, target, _paper_align in ROWS:
        cands = [(f, doc) for hay, f, doc in index if pat.lower() in hay]
        best = None
        newest_mtime = max((f.stat().st_mtime for f, _ in cands), default=0)
        # Prefer the largest scale, then the newest, the way align.py does. Sorting by
        # mtime alone let a reduced-scale diagnostic overwrite the full-scale result --
        # Qwen-3-Next reported 0.804x from a 16-seq attention A/B while its 1000-prompt
        # run sat in the same tree.
        def _rank(t):
            f, doc = t
            return (_scale(doc) or 0, f.stat().st_mtime)
        for rj, doc in sorted(cands, key=_rank):
            sp = speedups(doc)
            if sp:
                best = (mean(sp), alignment(doc), rj.stat().st_mtime, _scale(doc),
                        rj.stat().st_mtime < newest_mtime)
        if best is None:
            missing.append(row)
            # Distinguish "not run yet" from "ran but produced no comparison":
            # a bench can exit 0 with a failed reference side, leaving a result
            # file that has our throughput and no baseline.
            ran = bool(cands)
            note = "(RAN, NO BASELINE)" if ran else "(no result yet)"
            print(f"{cat:<12} {row:<20} {target:>6.2f} {'—':>7} {'—':>7}  {'—':<12}  {note}")
            continue
        got, al, mtime, scale, is_older = best
        ratio = got / target
        mark = "" if ratio >= 1 - a.tol else "  <-- BEHIND"
        if mark:
            behind.append((row, target, got))
        stamp = time.strftime("%m-%d %H:%M", time.localtime(mtime))
        # A result older than --after is from an earlier (e.g. reduced-scale
        # smoke) pass, not the run in progress. Reporting those as if they were
        # full-scale is an easy and invisible mistake.
        stale = " STALE" if a.after and stamp < a.after else ""
        # A reduced-scale diagnostic must not silently become the row's number:
        # compare.py picks the newest file that *has* a speedup, so when the
        # latest full run produced none (failed reference), an older/smaller run
        # wins by default. Surface both facts.
        if is_older:
            stale += " OLDER"
        if scale:
            stale += f" n={scale}"
        print(f"{cat:<12} {row:<20} {target:>6.2f} {got:>7.3f} {ratio:>7.2f}  "
              f"{stamp}{stale}  {al}{mark}")

    print(f"\n{len(ROWS) - len(missing)}/{len(ROWS)} rows have results; "
          f"{len(behind)} below {int((1 - a.tol) * 100)}% of paper")
    for row, t, g in behind:
        print(f"  BEHIND {row}: paper {t:.2f}x, B200 {g:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
