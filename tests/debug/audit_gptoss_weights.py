#!/usr/bin/env python3
"""Audit kb_nano GPT-OSS TP weight loading against vLLM's slicing rules."""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def _patch_tp(rank: int, tp: int) -> None:
    import kb_nano.infra.tp as tp_mod
    from kb_nano.tasks.baseline.L2 import gpt_oss_moe, parallel_embedding, parallel_linear

    def _size() -> int:
        return tp

    def _rank() -> int:
        return rank

    for mod in (tp_mod, gpt_oss_moe, parallel_embedding, parallel_linear):
        mod._tp_size = _size
        mod._tp_rank = _rank


def _iter_tensors(model_path: str):
    for sf_file in sorted(glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(sf_file, "pt", "cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def _assert_equal(report: list[str], name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        report.append(f"{name}: shape actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
        return
    if actual.dtype != expected.dtype:
        report.append(f"{name}: dtype actual={actual.dtype} expected={expected.dtype}")
        return
    if not torch.equal(actual, expected):
        if actual.is_floating_point():
            max_diff = (actual.float() - expected.float()).abs().max().item()
            report.append(f"{name}: values differ max_abs={max_diff}")
        else:
            mismatches = (actual != expected).sum().item()
            report.append(f"{name}: values differ mismatches={mismatches}")


def _expected_zero_tail(param: torch.Tensor, used: tuple[slice, ...]) -> torch.Tensor:
    mask = torch.ones(param.shape, dtype=torch.bool)
    mask[used] = False
    return param[mask]


def audit_rank(model_name: str, rank: int, tp: int, limit: int | None = None) -> list[str]:
    _patch_tp(rank, tp)

    from kb_nano.infra.weight_loader import _load_gpt_oss_weights
    from kb_nano.tasks.baseline.L4.gpt_oss import GptOssConfig, GptOssForCausalLM

    model_path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    cfg = GptOssConfig.from_pretrained(model_name)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = GptOssForCausalLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    _load_gpt_oss_weights(model, model_path)
    params = dict(model.named_parameters())

    report: list[str] = []
    seen: set[str] = set()
    expert_re = re.compile(
        r"(model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)_(blocks|scales|bias)$"
    )
    packed = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    I = cfg.intermediate_size // tp
    I_pad = math.ceil(I / 64) * 64
    H = cfg.hidden_size
    BLK = 32

    checked = 0
    for ckpt_name, tensor in _iter_tensors(model_path):
        if limit is not None and checked >= limit:
            break

        m = expert_re.match(ckpt_name)
        if m:
            prefix, proj, part = m.groups()
            base = prefix.replace("mlp.experts", "mlp")
            if proj == "gate_up_proj":
                suffix = {
                    "blocks": "w13_weight",
                    "scales": "w13_weight_scale",
                    "bias": "w13_bias",
                }[part]
                pname = f"{base}.{suffix}"
                param = params[pname].detach().cpu()
                start = 2 * rank * I
                width = 2 * I
                if part == "blocks":
                    loaded = tensor.reshape(cfg.num_local_experts, 2 * cfg.intermediate_size, -1)
                    expected = torch.zeros_like(param)
                    expected[:, :width, :].copy_(loaded[:, start:start + width, :].to(param.dtype))
                    used = (slice(None), slice(0, width), slice(None))
                elif part == "scales":
                    expected = torch.zeros_like(param)
                    expected[:, :width, :].copy_(tensor[:, start:start + width, :].to(param.dtype))
                    used = (slice(None), slice(0, width), slice(None))
                else:
                    expected = torch.zeros_like(param)
                    expected[:, :width].copy_(tensor[:, start:start + width].to(param.dtype))
                    used = (slice(None), slice(0, width))
                _assert_equal(report, pname, param, expected)
                if _expected_zero_tail(param, used).numel() and not torch.equal(
                    _expected_zero_tail(param, used),
                    torch.zeros_like(_expected_zero_tail(param, used)),
                ):
                    report.append(f"{pname}: padded tail is non-zero")
                seen.add(pname)
                checked += 1
                continue

            suffix = {
                "blocks": "w2_weight",
                "scales": "w2_weight_scale",
                "bias": "w2_bias",
            }[part]
            pname = f"{base}.{suffix}"
            param = params[pname].detach().cpu()
            expected = torch.zeros_like(param)
            if part == "blocks":
                loaded = tensor.reshape(cfg.num_local_experts, H, cfg.intermediate_size // 2)
                half = I // 2
                expected[:, :, :half].copy_(loaded[:, :, rank * half:rank * half + half].to(param.dtype))
                used = (slice(None), slice(None), slice(0, half))
            elif part == "scales":
                blocks = I // BLK
                expected[:, :, :blocks].copy_(tensor[:, :, rank * blocks:rank * blocks + blocks].to(param.dtype))
                used = (slice(None), slice(None), slice(0, blocks))
            else:
                if rank == 0:
                    expected.copy_(tensor.to(param.dtype))
                used = (slice(None), slice(None))
            _assert_equal(report, pname, param, expected)
            if part != "bias" and _expected_zero_tail(param, used).numel() and not torch.equal(
                _expected_zero_tail(param, used),
                torch.zeros_like(_expected_zero_tail(param, used)),
            ):
                report.append(f"{pname}: padded tail is non-zero")
            seen.add(pname)
            checked += 1
            continue

        mapped = ckpt_name
        if ckpt_name == "model.embed_tokens.weight":
            mapped = "model.embed_tokens.embedding_op.emb.weight"
            shard = tensor.shape[0] // tp
            expected = tensor.narrow(0, rank * shard, shard)
            param = params[mapped].detach().cpu()
            _assert_equal(report, mapped, param, expected.to(param.dtype))
            seen.add(mapped)
            checked += 1
            continue
        elif ckpt_name == "lm_head.weight":
            mapped = "lm_head.embedding_op.emb.weight"
            shard = tensor.shape[0] // tp
            expected = tensor.narrow(0, rank * shard, shard)
            param = params[mapped].detach().cpu()
            _assert_equal(report, mapped, param, expected.to(param.dtype))
            seen.add(mapped)
            checked += 1
            continue
        else:
            matched = False
            for orig, (packed_name, shard_id) in packed.items():
                if orig in mapped:
                    mapped = mapped.replace(orig, packed_name)
                    q_heads = cfg.num_attention_heads // tp
                    kv_heads = cfg.num_key_value_heads // tp
                    q = q_heads * cfg.head_dim
                    k = kv_heads * cfg.head_dim
                    if shard_id == "q":
                        expected = tensor.chunk(tp, 0)[rank]
                        offset = 0
                    elif shard_id == "k":
                        expected = tensor.chunk(tp, 0)[rank]
                        offset = q
                    else:
                        expected = tensor.chunk(tp, 0)[rank]
                        offset = q + k
                    param = params[mapped].detach().cpu()
                    target = param.narrow(0, offset, expected.shape[0])
                    _assert_equal(report, f"{mapped}[{shard_id}]", target, expected.to(target.dtype))
                    seen.add(mapped)
                    checked += 1
                    matched = True
                    break
            if matched:
                continue

            if "rotary_emb" in mapped or mapped not in params:
                continue

            param = params[mapped].detach().cpu()
            if mapped.endswith("o_proj.weight"):
                shard = param.shape[1]
                expected = tensor.narrow(1, rank * shard, shard)
            elif mapped.endswith("sinks"):
                heads = param.shape[0]
                expected = tensor.narrow(0, rank * heads, heads)
            else:
                expected = tensor
            _assert_equal(report, mapped, param, expected.to(param.dtype))
            seen.add(mapped)
            checked += 1

    missing = sorted(set(params) - seen)
    missing = [name for name in missing if "w13" not in name or limit is None]
    if limit is None and missing:
        report.append(f"unverified params: {missing[:20]} total={len(missing)}")

    del model
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ranks = range(args.tp) if args.rank is None else [args.rank]
    failed = False
    for rank in ranks:
        report = audit_rank(args.model, rank, args.tp, args.limit)
        if report:
            failed = True
            print(f"rank {rank}: FAIL ({len(report)} issues)")
            for line in report[:100]:
                print("  ", line)
        else:
            print(f"rank {rank}: PASS")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
