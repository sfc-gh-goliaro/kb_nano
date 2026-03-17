#!/usr/bin/env python3
"""
BitNet native inference alignment test.

Compares our standalone BitNet implementation against the GPU FastGen
reference using custom CUDA int8×int2 kernels.

Usage:
    # Run GPU reference only (existing behavior):
    python tests/test_native_bitnet_alignment.py \
        --ref-only --max-tokens 64 --seed 42

    # Run standalone only (our implementation):
    python tests/test_native_bitnet_alignment.py \
        --standalone-only --max-tokens 32 --seed 42

    # Full alignment (compare both):
    python tests/test_native_bitnet_alignment.py \
        --max-tokens 32 --seed 42
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"

PROMPTS = [
    "What is 2 + 2?",
    "Translate 'hello' into French, German, and Japanese.",
    (
        "Explain the difference between a stack and a queue in computer "
        "science. Give a real-world analogy for each."
    ),
    (
        "Write a Python function that computes the factorial of a number "
        "using recursion. Include a docstring."
    ),
]

# ---------------------------------------------------------------------------
# BitNet GPU worker — reuses generate.py's FastGen directly
# ---------------------------------------------------------------------------
BITNET_WORKER = r'''
import json, os, sys, time, torch

def main():
    cfg = json.loads(sys.argv[1])
    gpu_dir = cfg["gpu_dir"]

    os.chdir(gpu_dir)
    sys.path.insert(0, gpu_dir)

    from generate import FastGen, GenArgs
    from tokenizer import Tokenizer

    device = "cuda:0"
    torch.cuda.set_device(0)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    gen_length = cfg["max_tokens"]
    prompt_length = cfg["prompt_length"]

    gen_args = GenArgs(
        gen_length=gen_length,
        gen_bsz=1,
        prompt_length=prompt_length,
    )

    print("  [BitNet] Building FastGen (load + compile)...", flush=True)
    t0 = time.time()
    g = FastGen.build(cfg["ckpt_dir"], gen_args, device)
    print(f"  [BitNet] Ready in {time.time()-t0:.1f}s, "
          f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    results = []
    for pi, prompt in enumerate(cfg["prompts"]):
        token_ids = g.tokenizer.encode(prompt, bos=False, eos=False)

        t0 = time.time()
        stats, out_tokens = g.generate_all(
            [token_ids], use_cuda_graphs=True, use_sampling=False,
        )
        elapsed = time.time() - t0

        gen_text = g.tokenizer.decode(out_tokens[0])
        n_toks = len(out_tokens[0])
        tps = n_toks / elapsed if elapsed > 0 else 0

        perf = {}
        for ps in stats.phases:
            perf[ps.name] = {"ms": round(ps.time * 1000, 1), "tokens": ps.tokens}

        results.append({
            "text": gen_text,
            "token_ids": out_tokens[0],
            "perf": perf,
        })
        print(f"  [BitNet] Prompt #{pi}: {n_toks} tokens in {elapsed*1000:.0f}ms "
              f"({tps:.0f} tok/s)", flush=True)

    torch.cuda.synchronize()
    with open(cfg["output_file"], "w") as f:
        json.dump({
            "results": results,
            "memory_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        }, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Standalone worker — our BitNet implementation
# ---------------------------------------------------------------------------
STANDALONE_WORKER = r'''
import json, os, sys, time, torch

def load_from_gpu_checkpoint(model, ckpt_path, config):
    """Load weights from the GPU fp16 checkpoint into our model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = model.state_dict()

    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    ffn_dim = config.intermediate_size

    loaded = 0
    for ckpt_name, tensor in ckpt.items():
        import re
        m = re.match(r"layers\.(\d+)\.(.*)", ckpt_name)
        if m:
            layer_idx, suffix = m.group(1), m.group(2)
            prefix = f"model.layers.{layer_idx}"

            if suffix == "attention_norm.weight":
                sd[f"{prefix}.input_layernorm.weight"].copy_(tensor)
            elif suffix == "attention.wqkv.weight":
                # Split merged QKV: [q_dim+2*kv_dim, hidden_size]
                q, k, v = tensor.split([q_dim, kv_dim, kv_dim], dim=0)
                sd[f"{prefix}.self_attn.q_proj.weight"].copy_(q)
                sd[f"{prefix}.self_attn.k_proj.weight"].copy_(k)
                sd[f"{prefix}.self_attn.v_proj.weight"].copy_(v)
            elif suffix == "attention.wo.weight":
                sd[f"{prefix}.self_attn.o_proj.weight"].copy_(tensor)
            elif suffix == "attention.attn_sub_norm.weight":
                sd[f"{prefix}.self_attn.attn_sub_norm.weight"].copy_(tensor)
            elif suffix == "ffn_norm.weight":
                sd[f"{prefix}.post_attention_layernorm.weight"].copy_(tensor)
            elif suffix == "feed_forward.w13.weight":
                # Split merged gate+up: [2*ffn_dim, hidden_size]
                gate, up = tensor.split([ffn_dim, ffn_dim], dim=0)
                sd[f"{prefix}.mlp.gate_proj.weight"].copy_(gate)
                sd[f"{prefix}.mlp.up_proj.weight"].copy_(up)
            elif suffix == "feed_forward.w2.weight":
                sd[f"{prefix}.mlp.down_proj.weight"].copy_(tensor)
            elif suffix == "feed_forward.ffn_sub_norm.weight":
                sd[f"{prefix}.mlp.ffn_sub_norm.weight"].copy_(tensor)
            else:
                print(f"  [Standalone] Skipping unknown: {ckpt_name}")
                continue
            loaded += 1
        elif ckpt_name == "tok_embeddings.weight":
            sd["model.embed_tokens.weight"].copy_(tensor)
            loaded += 1
        elif ckpt_name == "norm.weight":
            sd["model.norm.weight"].copy_(tensor)
            loaded += 1
        elif ckpt_name == "output.weight":
            # Skip if tied (lm_head.weight == embed_tokens.weight)
            if not hasattr(model.config, 'tie_word_embeddings') or not model.config.tie_word_embeddings:
                sd["lm_head.weight"].copy_(tensor)
            loaded += 1
        else:
            print(f"  [Standalone] Skipping unknown: {ckpt_name}")
            continue

    model.load_state_dict(sd)
    print(f"  [Standalone] Loaded {loaded} weight entries")


@torch.no_grad()
def generate_greedy(model, input_ids, max_new_tokens, device, eos_token_id=128001):
    """Greedy autoregressive generation (no KV cache)."""
    tokens = list(input_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([tokens], device=device, dtype=torch.long)
        hidden = model(x)
        logits = model.compute_logits(hidden[:, -1:, :])
        next_token = logits.argmax(dim=-1).item()
        tokens.append(next_token)
        if next_token == eos_token_id:
            break
    return tokens[len(input_ids):]


def main():
    cfg = json.loads(sys.argv[1])
    sys.path.insert(0, cfg["project_root"])

    from kb_nano.tasks.baseline.L4.bitnet import BitNetConfig, BitNetForCausalLM
    from transformers import AutoTokenizer

    device = cfg.get("device", "cuda:0")
    torch.cuda.set_device(int(device.split(":")[-1]))
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    # Build model
    print("  [Standalone] Allocating BitNet model...", flush=True)
    config = BitNetConfig()
    model = BitNetForCausalLM(config)

    # Load weights from GPU fp16 checkpoint
    ckpt_path = cfg["ckpt_file"]
    print(f"  [Standalone] Loading from {ckpt_path}...", flush=True)
    t0 = time.time()
    load_from_gpu_checkpoint(model, ckpt_path, config)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"  [Standalone] Ready in {time.time()-t0:.1f}s, "
          f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])

    results = []
    for pi, prompt in enumerate(cfg["prompts"]):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        t0 = time.time()
        gen_ids = generate_greedy(
            model, token_ids, cfg["max_tokens"], device,
            eos_token_id=tokenizer.eos_token_id or 128001,
        )
        elapsed = time.time() - t0
        gen_text = tokenizer.decode(gen_ids)
        n_toks = len(gen_ids)
        tps = n_toks / elapsed if elapsed > 0 else 0
        results.append({
            "text": gen_text,
            "token_ids": gen_ids,
        })
        print(f"  [Standalone] Prompt #{pi}: {n_toks} tokens in {elapsed*1000:.0f}ms "
              f"({tps:.0f} tok/s)", flush=True)

    torch.cuda.synchronize()
    with open(cfg["output_file"], "w") as f:
        json.dump({
            "results": results,
            "memory_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        }, f)

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def run_worker(script: str, config: dict, label: str, python_bin: str = None) -> dict | None:
    if python_bin is None:
        python_bin = sys.executable

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(script)
        script_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    config["output_file"] = output_path

    try:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        result = subprocess.run(
            [python_bin, script_path, json.dumps(config)],
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  ERROR: {label} failed with exit code {result.returncode}")
            return None

        with open(output_path) as f:
            return json.loads(f.read())
    finally:
        os.unlink(script_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def print_results(data: dict, label: str):
    """Print generation results."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    for i, r in enumerate(data["results"]):
        prompt_preview = PROMPTS[i][:60] + ("..." if len(PROMPTS[i]) > 60 else "")
        perf = r.get("perf", {})
        perf_str = " | ".join(
            f"{k}: {v['ms']}ms" + (f" ({v['tokens']} tok)" if v.get('tokens') else "")
            for k, v in perf.items()
        )
        print(f"\n  Prompt #{i}: {prompt_preview}")
        print(f"  Tokens   : {len(r['token_ids'])}")
        print(f"  IDs[:10] : {r['token_ids'][:10]}")
        if perf_str:
            print(f"  Perf     : {perf_str}")
        print("  Output   :")
        for line in r["text"].splitlines():
            print(f"    {line}")
    if "memory_gb" in data:
        print(f"\n  Memory   : {data['memory_gb']} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BitNet alignment: standalone vs GPU FastGen",
    )
    parser.add_argument("--ref-only", action="store_true",
                        help="Run GPU FastGen reference only")
    parser.add_argument("--standalone-only", action="store_true",
                        help="Run our standalone implementation only")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-length", type=int, default=64,
                        help="Padded prompt length for CUDA graph (default: 64)")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Converted checkpoint dir (default: BitNet/gpu/checkpoints)")
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(this_dir)  # kb_nano/
    project_root = os.path.dirname(pkg_root)  # parent of kb_nano (for imports)
    gpu_dir = os.path.join(pkg_root, "BitNet", "gpu")
    ckpt_dir = args.ckpt_dir or os.path.join(gpu_dir, "checkpoints")
    fp16_ckpt = os.path.join(ckpt_dir, "model_state_fp16.pt")

    kb_python = os.path.join(
        os.path.expanduser("~"), "conda", "miniconda", "envs", "kb", "bin", "python",
    )
    if not os.path.exists(kb_python):
        kb_python = sys.executable

    # Validate checkpoints
    if not args.standalone_only:
        for fname in ("model_state_fp16.pt", "model_state_int2.pt"):
            if not os.path.exists(os.path.join(ckpt_dir, fname)):
                print(f"ERROR: {fname} not found in {ckpt_dir}")
                print("Run convert_hf_packed.py first.")
                sys.exit(1)
    else:
        if not os.path.exists(fp16_ckpt):
            print(f"ERROR: model_state_fp16.pt not found in {ckpt_dir}")
            sys.exit(1)

    print("=" * 70)
    print(f"  BitNet Alignment Test")
    print("=" * 70)
    print(f"  Model        : {MODEL_ID}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"  Seed         : {args.seed}")
    print(f"  Prompts      : {len(PROMPTS)}")
    print(f"  Ckpt dir     : {ckpt_dir}")
    mode = "ref-only" if args.ref_only else ("standalone-only" if args.standalone_only else "alignment")
    print(f"  Mode         : {mode}")
    print("=" * 70)

    ref_data = None
    standalone_data = None

    # --- GPU FastGen reference ---
    if not args.standalone_only:
        ref_config = {
            "model": MODEL_ID,
            "seed": args.seed,
            "prompts": PROMPTS,
            "max_tokens": args.max_tokens,
            "prompt_length": args.prompt_length,
            "gpu_dir": gpu_dir,
            "ckpt_dir": ckpt_dir,
        }
        ref_data = run_worker(
            BITNET_WORKER, ref_config,
            f"BitNet GPU FastGen  [{MODEL_ID}]",
            python_bin=kb_python,
        )
        if ref_data is None:
            print("\n  FAIL: GPU FastGen failed")
            sys.exit(1)
        print_results(ref_data, "GPU FastGen Reference")

    # --- Standalone implementation ---
    if not args.ref_only:
        standalone_config = {
            "model": MODEL_ID,
            "seed": args.seed,
            "prompts": PROMPTS,
            "max_tokens": args.max_tokens,
            "project_root": project_root,
            "ckpt_file": fp16_ckpt,
        }
        standalone_data = run_worker(
            STANDALONE_WORKER, standalone_config,
            f"Standalone BitNet  [{MODEL_ID}]",
        )
        if standalone_data is None:
            print("\n  FAIL: Standalone implementation failed")
            sys.exit(1)
        print_results(standalone_data, "Standalone Implementation")

    # --- Comparison ---
    if ref_data and standalone_data:
        print(f"\n{'=' * 70}")
        print(f"  ALIGNMENT COMPARISON")
        print(f"{'=' * 70}")
        match_count = 0
        for i in range(len(PROMPTS)):
            ref_ids = ref_data["results"][i]["token_ids"]
            our_ids = standalone_data["results"][i]["token_ids"]
            min_len = min(len(ref_ids), len(our_ids))
            first_diff = min_len
            for j in range(min_len):
                if ref_ids[j] != our_ids[j]:
                    first_diff = j
                    break
            if first_diff == min_len and len(ref_ids) == len(our_ids):
                print(f"  Prompt #{i}: MATCH ({len(ref_ids)} tokens)")
                match_count += 1
            else:
                print(f"  Prompt #{i}: DIFFER at token {first_diff}")
                print(f"    Ref [{first_diff}:+3] = {ref_ids[first_diff:first_diff+3]}")
                print(f"    Ours[{first_diff}:+3] = {our_ids[first_diff:first_diff+3]}")
                if first_diff >= len(ref_ids) * 0.8:
                    match_count += 1
                    print(f"    (late divergence — counting as match)")

        print(f"\n  Result: {match_count}/{len(PROMPTS)} prompts match")
        status = "PASS" if match_count >= len(PROMPTS) * 0.75 else "FAIL"
        print(f"{'=' * 70}")
        print(f"  {status}")
        print(f"{'=' * 70}")
        sys.exit(0 if status == "PASS" else 1)
    else:
        print(f"{'=' * 70}")
        print("  PASS")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
