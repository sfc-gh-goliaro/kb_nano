#!/usr/bin/env python3
"""Debug DeepSeek-V3.2 alignment at operator level: kb-nano vs vLLM.

Runs both engines in separate subprocesses. vLLM dumps operator-level tensors
via VLLM_DEBUG_DUMP_DIR env var (patched into installed vLLM source).
kb-nano dumps via monkey-patched forward methods.

Usage:
    PYTHONUNBUFFERED=1 python tests/debug/debug_deepseek_alignment.py
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from random import randint, seed as rseed

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_TESTS_DIR = _THIS_DIR.parent
_PACKAGE_DIR = _TESTS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))
from kb_nano.bench.utils.worker import run_worker


# ═══════════════════════════════════════════════════════════════════════
# vLLM worker — uses VLLM_DEBUG_DUMP_DIR for tensor dumping
# ═══════════════════════════════════════════════════════════════════════
VLLM_DUMP_WORKER = r'''
import json, os, sys, time, torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    dump_dir = cfg["dump_dir"]
    os.makedirs(dump_dir, exist_ok=True)
    num_layers = cfg.get("num_layers")

    # Set dump dir BEFORE LLM init so the EngineCore subprocess inherits it.
    # Dumping is gated by a .dump_active signal file (written after warmup).
    os.environ["VLLM_DEBUG_DUMP_DIR"] = dump_dir

    from vllm import LLM, SamplingParams

    load_fmt = "auto" if num_layers is not None else "fastsafetensors"
    llm_kwargs = dict(
        model=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=True,
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg["max_model_len"],
        enable_prefix_caching=False,
        load_format=load_fmt,
    )
    if num_layers is not None:
        llm_kwargs["hf_overrides"] = {"num_hidden_layers": num_layers}

    llm = LLM(**llm_kwargs)

    # Warmup (no dumping yet — signal file not present)
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    print("  vLLM: warmup done", flush=True)

    # Activate dumping via signal file
    with open(os.path.join(dump_dir, ".dump_active"), "w") as f:
        f.write("1")

    prompt_ids = cfg["prompt_token_ids"]
    outputs = llm.generate(
        [dict(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0.0, max_tokens=1),
    )

    # Remove signal file
    os.remove(os.path.join(dump_dir, ".dump_active"))

    gen_ids = list(outputs[0].outputs[0].token_ids)

    torch.save(torch.tensor(gen_ids), os.path.join(dump_dir, "gen_ids.pt"))

    n_files = len([f for f in os.listdir(dump_dir) if f.endswith(".pt")])
    with open(cfg["output_file"], "w") as f:
        json.dump({"status": "ok", "generated_ids": gen_ids,
                    "num_tensors": n_files}, f)
    print(f"  vLLM: saved {n_files} tensor files to {dump_dir}", flush=True)

if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════
# kb-nano worker — monkey-patches forward to dump operator tensors
# ═══════════════════════════════════════════════════════════════════════
KB_NANO_DUMP_WORKER = r'''
import json, os, sys, time, torch

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    if cfg.get("num_layers") is not None:
        os.environ["KB_NANO_NUM_LAYERS"] = str(cfg["num_layers"])

    dump_dir = cfg["dump_dir"]
    os.makedirs(dump_dir, exist_ok=True)

    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=cfg["model"],
        seed=cfg["seed"],
        enforce_eager=True,
        tensor_parallel_size=cfg["tp"],
    )
    if "gpu_memory_utilization" in cfg:
        engine_kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if "max_model_len" in cfg:
        engine_kwargs["max_model_len"] = cfg["max_model_len"]
    engine = LlamaEngine(**engine_kwargs)

    # Warmup
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=1))
    print("  kb-nano: warmup done", flush=True)

    model = engine.model_runner.model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        inner = model.model
    else:
        inner = model

    layers = inner.layers
    num_l = len(layers)
    print(f"  kb-nano: {type(model).__name__}, {num_l} layers", flush=True)

    _saves = {}
    def _save(name, t):
        _saves[name] = t.detach().float().cpu()

    for li in range(num_l):
        layer = layers[li]
        attn = layer.self_attn

        def make_patched_attn(attn_self, layer_idx):
            def patched_attn_forward(positions, hidden_states):
                from kb_nano.infra.context import get_context
                N = hidden_states.shape[0]
                ctx = get_context()
                _save(f"L{layer_idx}_mla_input", hidden_states)

                if attn_self.q_a_proj is not None:
                    q_c = attn_self.q_a_proj(hidden_states)
                    _save(f"L{layer_idx}_q_c_pre_norm", q_c)
                    q_c = attn_self.q_a_layernorm(q_c)
                    _save(f"L{layer_idx}_q_c_post_norm", q_c)
                    q = attn_self.q_b_proj(q_c).view(
                        N, attn_self.num_local_heads, attn_self.qk_head_dim)
                    _save(f"L{layer_idx}_q_b_proj_out",
                          q.view(N, -1))
                else:
                    q = attn_self.q_proj(hidden_states).view(
                        N, attn_self.num_local_heads, attn_self.qk_head_dim)
                    q_c = None

                q_nope, q_pe = q.split(
                    [attn_self.qk_nope_head_dim, attn_self.qk_rope_head_dim],
                    dim=-1)

                latent = attn_self.kv_a_proj_with_mqa(hidden_states)
                kv_a, k_pe = latent.split(
                    [attn_self.kv_lora_rank, attn_self.qk_rope_head_dim], dim=-1)
                _save(f"L{layer_idx}_kv_lora", latent)

                kv_c_normed = attn_self.kv_a_layernorm(kv_a)
                _save(f"L{layer_idx}_kv_c_normed", kv_c_normed)
                _save(f"L{layer_idx}_k_pe_pre_rope", k_pe)

                q_pe_flat = q_pe.reshape(N, attn_self.num_local_heads * attn_self.qk_rope_head_dim)
                k_pe_flat = k_pe
                q_pe_flat, k_pe_flat = attn_self.rotary_emb(positions, q_pe_flat, k_pe_flat)
                q_pe = q_pe_flat.view(N, attn_self.num_local_heads, attn_self.qk_rope_head_dim)
                k_pe = k_pe_flat.view(N, attn_self.qk_rope_head_dim)

                _save(f"L{layer_idx}_q_pe_post_rope", q_pe)
                _save(f"L{layer_idx}_k_pe_post_rope", k_pe)
                _save(f"L{layer_idx}_q_nope", q_nope)

                q[..., attn_self.qk_nope_head_dim:] = q_pe

                kv_cache = attn_self.k_cache
                if kv_cache.numel():
                    attn_self._store_kvcache(kv_c_normed, k_pe, kv_cache, ctx.slot_mapping)

                if attn_self.indexer is not None and attn_self._topk_indices_buffer is not None:
                    q_c_for_idx = q_c if q_c is not None else attn_self.q_a_layernorm(attn_self.q_a_proj(hidden_states))
                    attn_self.indexer(hidden_states, q_c_for_idx, positions, attn_self._topk_indices_buffer)
                    _save(f"L{layer_idx}_topk_indices", attn_self._topk_indices_buffer[:N].float())

                attn_self._extract_absorption_weights()
                ql_nope = torch.einsum('bhd,hdc->bhc', q_nope, attn_self._w_uk)
                q_absorbed = torch.empty(
                    N, attn_self.num_local_heads,
                    attn_self.kv_lora_rank + attn_self.qk_rope_head_dim,
                    dtype=ql_nope.dtype, device=ql_nope.device)
                q_absorbed[..., :attn_self.kv_lora_rank] = ql_nope
                q_absorbed[..., attn_self.kv_lora_rank:] = q_pe

                if ctx.is_mixed:
                    attn_output = attn_self._forward_mixed(
                        q_absorbed, kv_c_normed, k_pe, kv_cache, ctx, N)
                elif ctx.is_prefill:
                    attn_output = attn_self._forward_prefill(
                        q_absorbed, kv_c_normed, k_pe, kv_cache, ctx, N)
                else:
                    attn_output = attn_self._forward_decode(
                        q_absorbed, kv_cache, ctx, N)
                _save(f"L{layer_idx}_attn_out_pre_oproj", attn_output)

                o_out = attn_self.o_proj(attn_output)
                _save(f"L{layer_idx}_o_proj_out", o_out)
                return o_out

            return patched_attn_forward

        attn.forward = make_patched_attn(attn, li)

        def make_patched_layer(layer_self, layer_idx):
            def patched_layer_forward(positions, hidden_states, residual):
                _save(f"L{layer_idx}_layer_input_hidden", hidden_states)
                if residual is not None:
                    _save(f"L{layer_idx}_layer_input_residual", residual)

                if residual is None:
                    hidden_states, residual = (
                        layer_self.input_layernorm(hidden_states), hidden_states)
                else:
                    hidden_states, residual = layer_self.input_layernorm(
                        hidden_states, residual)
                _save(f"L{layer_idx}_input_ln_out", hidden_states)

                hidden_states = layer_self.self_attn(positions, hidden_states)
                _save(f"L{layer_idx}_attn_full_out", hidden_states)

                hidden_states, residual = layer_self.post_attention_layernorm(
                    hidden_states, residual)
                _save(f"L{layer_idx}_post_attn_ln_out", hidden_states)

                hidden_states = layer_self.mlp(hidden_states)
                _save(f"L{layer_idx}_mlp_out", hidden_states)

                return hidden_states, residual

            return patched_layer_forward

        layer.forward = make_patched_layer(layer, li)

        from kb_nano.tasks.baseline.L2.deepseek_moe import DeepSeekMoE
        if isinstance(layer.mlp, DeepSeekMoE):
            moe = layer.mlp
            def make_moe_hooks(layer_idx, moe_ref):
                def gate_hook(module, args, output):
                    _save(f"L{layer_idx}_moe_router_logits", output)
                def shared_hook(module, args, output):
                    _save(f"L{layer_idx}_moe_shared_out", output.clone())
                def shared_gu_hook(module, args, output):
                    _save(f"L{layer_idx}_moe_shared_gu_out", output.clone())
                def shared_act_hook(module, args, output):
                    _save(f"L{layer_idx}_moe_shared_act_out", output.clone())
                def shared_dp_hook(module, args, output):
                    _save(f"L{layer_idx}_moe_shared_dp_out", output.clone())
                return gate_hook, shared_hook, shared_gu_hook, shared_act_hook, shared_dp_hook
            gh, sh, sguh, sah, sdph = make_moe_hooks(li, moe)
            moe.gate.register_forward_hook(gh)
            if moe.shared_experts is not None:
                se = moe.shared_experts
                se.register_forward_hook(sh)
                se.gate_up_proj.register_forward_hook(sguh)
                se.act_fn.register_forward_hook(sah)
                se.down_proj.register_forward_hook(sdph)

    # Embedding hook
    def embed_hook(module, args, output):
        _save("embed", output)
    eh = inner.embed_tokens.register_forward_hook(embed_hook)

    prompt_ids = cfg["prompt_token_ids"]
    engine.block_manager.reset()
    torch.cuda.synchronize()
    outputs = engine.generate(
        [prompt_ids],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
    )
    gen_ids = outputs[0].token_ids
    eh.remove()

    for name, tensor in _saves.items():
        torch.save(tensor, os.path.join(dump_dir, f"{name}.pt"))
    torch.save(torch.tensor(gen_ids), os.path.join(dump_dir, "gen_ids.pt"))

    n_files = len(_saves) + 1
    with open(cfg["output_file"], "w") as f:
        json.dump({"status": "ok", "generated_ids": list(gen_ids),
                    "num_tensors": n_files}, f)
    print(f"  kb-nano: saved {n_files} tensor files to {dump_dir}", flush=True)

    del engine

if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════

def compare(name, a, b, rtol=1e-2, atol=5e-3):
    a_f, b_f = a.float(), b.float()
    if a_f.shape != b_f.shape:
        print(f"  [SHAPE] {name}: kb={list(a.shape)} vs vllm={list(b.shape)}")
        n = min(a_f.shape[0], b_f.shape[0])
        a_f, b_f = a_f[:n], b_f[:n]
    diff = (a_f - b_f).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        a_f.reshape(1, -1), b_f.reshape(1, -1)).item()
    close = torch.allclose(a_f, b_f, rtol=rtol, atol=atol)
    tag = "OK" if close else "MISMATCH"
    print(f"  [{tag}] {name}: max={max_d:.6e}  mean={mean_d:.6e}  "
          f"cos={cos:.8f}  |kb|={a_f.norm():.4f}  |vl|={b_f.norm():.4f}")
    return max_d, cos


LAYER_OPS = [
    "layer_input_hidden",
    "layer_input_residual",
    "input_ln_out",
    "mla_input",
    "q_c_pre_norm",
    "kv_lora",
    "q_c_post_norm",
    "q_b_proj_out",
    "kv_c_normed",
    "k_pe_pre_rope",
    "q_nope",
    "q_pe_post_rope",
    "k_pe_post_rope",
    "topk_indices",
    "attn_out_pre_oproj",
    "o_proj_out",
    "attn_full_out",
    "post_attn_ln_out",
    "moe_router_logits",
    "moe_topk_weights",
    "moe_topk_ids",
    "moe_shared_gu_out",
    "moe_shared_act_out",
    "moe_shared_dp_out",
    "moe_shared_out",
    "moe_routed_out_pre_scale",
    "moe_routed_out_post_scale",
    "mlp_out",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-mem", type=float, default=0.45)
    args = parser.parse_args()

    rseed(args.seed); np.random.seed(args.seed)
    prompt_ids = [randint(100, 10000) for _ in range(args.seq_len)]
    max_model_len = args.seq_len + 64

    vllm_dir = tempfile.mkdtemp(prefix="vllm_dump_")
    kb_dir = tempfile.mkdtemp(prefix="kb_dump_")

    print("=" * 70)
    print("  DeepSeek-V3.2 Operator-Level Alignment Debug")
    print("=" * 70)
    print(f"  Layers: {args.num_layers}  SeqLen: {args.seq_len}  Seed: {args.seed}")
    print(f"  vLLM dir: {vllm_dir}")
    print(f"  kb   dir: {kb_dir}")
    print("=" * 70, flush=True)

    # ── Run vLLM ────────────────────────────────────────────────────
    r1 = run_worker(VLLM_DUMP_WORKER, {
        "model": args.model, "tp": args.tp, "seed": args.seed,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "num_layers": args.num_layers,
        "prompt_token_ids": prompt_ids,
        "dump_dir": vllm_dir,
    }, f"vLLM ({args.num_layers}L, seq={args.seq_len})", timeout=3600)

    if r1 is None:
        print("ERROR: vLLM failed"); sys.exit(1)

    # ── Run kb-nano ─────────────────────────────────────────────────
    r2 = run_worker(KB_NANO_DUMP_WORKER, {
        "model": args.model, "tp": args.tp, "seed": args.seed,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "num_layers": args.num_layers,
        "prompt_token_ids": prompt_ids,
        "dump_dir": kb_dir,
        "project_root": str(_PROJECT_ROOT),
        "package_name": _PACKAGE_DIR.name,
    }, f"kb-nano ({args.num_layers}L, seq={args.seq_len})", timeout=3600)

    if r2 is None:
        print("ERROR: kb-nano failed"); sys.exit(1)

    # ── Compare ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OPERATOR-LEVEL COMPARISON")
    print("=" * 70)

    v_ids = r1.get("generated_ids", [])
    k_ids = r2.get("generated_ids", [])
    print(f"\n  Tokens — vLLM: {v_ids}  kb: {k_ids}  match={v_ids==k_ids}")

    # Embedding
    vp = os.path.join(vllm_dir, "embed.pt")
    kp = os.path.join(kb_dir, "embed.pt")
    if os.path.exists(vp) and os.path.exists(kp):
        print("\n--- Embedding ---")
        compare("embed", torch.load(kp, weights_only=True),
                torch.load(vp, weights_only=True))

    first_bad = None
    for li in range(args.num_layers):
        print(f"\n{'─'*50} Layer {li} {'─'*50}")
        for op in LAYER_OPS:
            name = f"L{li}_{op}"
            vp = os.path.join(vllm_dir, f"{name}.pt")
            kp = os.path.join(kb_dir, f"{name}.pt")
            if os.path.exists(vp) and os.path.exists(kp):
                vt = torch.load(vp, weights_only=True)
                kt = torch.load(kp, weights_only=True)
                _, cos = compare(name, kt, vt)
                if cos < 0.99 and first_bad is None:
                    first_bad = name
            elif os.path.exists(vp) or os.path.exists(kp):
                which = "kb only" if os.path.exists(kp) else "vllm only"
                print(f"  [SKIP] {name}: {which}")

    # Summary
    print(f"\n{'='*70}")
    print("  DIAGNOSIS")
    print(f"{'='*70}")
    if first_bad:
        print(f"  First operator with cos_sim < 0.99: {first_bad}")
        parts = first_bad.split("_", 1)
        layer = parts[0]
        op = parts[1] if len(parts) > 1 else "?"
        print(f"  => Bug is in {layer}, operator '{op}'")
    else:
        print("  All operators match well (cos_sim >= 0.99)")
    print(f"{'='*70}")

    # Don't clean up — user may want to inspect
    print(f"\n  Dump dirs preserved:")
    print(f"    vLLM:    {vllm_dir}")
    print(f"    kb-nano: {kb_dir}")


if __name__ == "__main__":
    main()
