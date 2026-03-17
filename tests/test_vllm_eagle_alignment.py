#!/usr/bin/env python3
"""
Reference-only vLLM EAGLE explicit-forward alignment probe.

Runs a hand-written speculative loop on 4 fixed prompts:
1) explicit draft step via EAGLE `forward(...)`
2) explicit target verification (greedy)
3) acceptance-rate report
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


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


VLLM_EAGLE_WORKER = r'''
import inspect
import json
import os
import sys
from collections import deque

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.forward_context import set_forward_context
from vllm import LLM


def _get_by_chain(root, chain):
    cur = root
    for name in chain:
        if cur is None or not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def _find_speculator(llm):
    chains = [
        ["llm_engine", "model_executor", "driver_worker", "model_runner", "speculator"],
        ["llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "speculator"],
        ["llm_engine", "model_runner", "speculator"],
        ["llm_engine", "model_executor", "driver_worker", "model_runner", "drafter"],
        ["llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "drafter"],
        ["llm_engine", "model_runner", "drafter"],
        ["llm_engine", "engine_core", "model_executor", "driver_worker", "model_runner", "drafter"],
        ["llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "drafter"],
        ["llm_engine", "engine_core", "model_runner", "drafter"],
    ]
    for chain in chains:
        obj = _get_by_chain(llm, chain)
        if obj is not None:
            return obj, ".".join(chain)

    q = deque([("llm", llm, 0)])
    seen = set()
    while q:
        path, obj, depth = q.popleft()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        cls_name = type(obj).__name__.lower()
        method_name = str(getattr(obj, "method", "")).lower()
        if hasattr(obj, "model") and (
            "speculator" in cls_name
            or "drafter" in cls_name
            or "draft" in cls_name
            or "eagle" in cls_name
            or "eagle" in method_name
            or hasattr(obj, "propose")
        ):
            return obj, path

        if depth >= 8:
            continue

        for name in dir(obj):
            if name.startswith("_"):
                continue
            if (
                "spec" not in name
                and "draft" not in name
                and "eagle" not in name
                and "engine" not in name
                and "core" not in name
                and "executor" not in name
                and "worker" not in name
                and "runner" not in name
                and "model" not in name
            ):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if callable(child):
                continue
            q.append((f"{path}.{name}", child, depth + 1))

    return None, None


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for y in x:
            t = _first_tensor(y)
            if t is not None:
                return t
    if hasattr(x, "logits") and torch.is_tensor(x.logits):
        return x.logits
    if hasattr(x, "hidden_states"):
        hs = getattr(x, "hidden_states")
        if torch.is_tensor(hs):
            return hs
        if isinstance(hs, (list, tuple)):
            for y in reversed(hs):
                if torch.is_tensor(y):
                    return y
    if hasattr(x, "last_hidden_state") and torch.is_tensor(x.last_hidden_state):
        return x.last_hidden_state
    return None


def _target_next_and_hidden(model, context_ids, device):
    input_ids = torch.tensor([context_ids], dtype=torch.int64, device=device)
    out = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits[:, -1, :]
    hidden = out.hidden_states[-1][:, -1, :]
    next_token = int(torch.argmax(logits, dim=-1).item())
    return next_token, hidden


def _call_eagle_forward(
    eagle_model,
    vllm_config,
    input_ids,
    positions,
    hidden_states,
):
    sig = getattr(eagle_model, "_codex_forward_sig", None)
    if sig is None:
        sig = inspect.signature(eagle_model.forward)
        setattr(eagle_model, "_codex_forward_sig", sig)
    params = sig.parameters

    kwargs = {}
    if "input_ids" in params:
        kwargs["input_ids"] = input_ids
    elif "token_ids" in params:
        kwargs["token_ids"] = input_ids
    else:
        raise RuntimeError("EAGLE forward missing input_ids/token_ids argument.")

    if "positions" in params:
        kwargs["positions"] = positions
    elif "position_ids" in params:
        kwargs["position_ids"] = positions.view(1, -1)

    if "hidden_states" in params:
        kwargs["hidden_states"] = hidden_states
    elif "inputs_embeds" in params:
        kwargs["inputs_embeds"] = hidden_states.unsqueeze(1)
    else:
        raise RuntimeError("EAGLE forward missing hidden_states/inputs_embeds argument.")

    num_tokens = int(input_ids.numel())
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        slot_mapping={},
    ):
        return eagle_model(**kwargs)


def _draft_next_and_hidden(
    eagle_model,
    vllm_config,
    prev_token_id,
    prev_position,
    hidden_states,
):
    try:
        ref_param = next(eagle_model.parameters())
    except StopIteration as exc:
        raise RuntimeError("EAGLE model has no parameters.") from exc

    device = ref_param.device
    dtype = ref_param.dtype

    hs = hidden_states
    if hs.dim() == 3:
        hs = hs[:, -1, :]
    hs = hs.to(device=device, dtype=dtype, non_blocking=True).contiguous()

    input_ids = torch.tensor([int(prev_token_id)], dtype=torch.int64, device=device)
    positions = torch.tensor([int(prev_position)], dtype=torch.int64, device=device)

    out = _call_eagle_forward(
        eagle_model=eagle_model,
        vllm_config=vllm_config,
        input_ids=input_ids,
        positions=positions,
        hidden_states=hs,
    )
    latent = _first_tensor(out)
    if latent is None:
        raise RuntimeError("Could not extract tensor output from EAGLE forward.")

    if latent.dim() == 3:
        latent_last = latent[:, -1, :]
    elif latent.dim() == 2:
        latent_last = latent
    else:
        latent_last = latent.reshape(1, -1)

    logits = None
    if hasattr(eagle_model, "compute_logits"):
        try:
            logits = eagle_model.compute_logits(latent_last)
        except Exception:
            logits = eagle_model.compute_logits(latent_last.unsqueeze(1))
    if logits is None:
        if hasattr(out, "logits") and torch.is_tensor(out.logits):
            logits = out.logits
        else:
            raise RuntimeError(
                "Could not compute EAGLE logits. Missing compute_logits and out.logits."
            )

    if logits.dim() == 3:
        logits = logits[:, -1, :]
    elif logits.dim() != 2:
        logits = logits.reshape(1, -1)

    next_token = int(torch.argmax(logits, dim=-1).item())
    return next_token, latent_last


def _spec_decode_prompt(
    prompt,
    tokenizer,
    target_model,
    eagle_model,
    eagle_vllm_config,
    num_speculative_tokens,
    max_tokens,
    accepted_per_pos,
):
    encoded = tokenizer(prompt, return_tensors="pt")
    context_ids = encoded["input_ids"][0].tolist()
    if not context_ids:
        if tokenizer.bos_token_id is not None:
            context_ids = [int(tokenizer.bos_token_id)]
        elif tokenizer.eos_token_id is not None:
            context_ids = [int(tokenizer.eos_token_id)]
        else:
            context_ids = [1]

    target_device = next(target_model.parameters()).device
    eos_token_id = tokenizer.eos_token_id

    generated_ids = []
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0

    while len(generated_ids) < max_tokens:
        draft_len = min(num_speculative_tokens, max_tokens - len(generated_ids))
        if draft_len <= 0:
            break

        num_drafts += 1
        accepted_this_round = 0
        rejected = False

        target_next, target_hidden = _target_next_and_hidden(
            target_model,
            context_ids,
            target_device,
        )
        prev_token = int(context_ids[-1])
        prev_pos = len(context_ids) - 1
        draft_hidden = None

        for j in range(draft_len):
            hidden_in = target_hidden if j == 0 else draft_hidden
            draft_next, draft_hidden = _draft_next_and_hidden(
                eagle_model=eagle_model,
                vllm_config=eagle_vllm_config,
                prev_token_id=prev_token,
                prev_position=prev_pos,
                hidden_states=hidden_in,
            )
            num_draft_tokens += 1

            if draft_next == target_next:
                context_ids.append(draft_next)
                generated_ids.append(draft_next)
                num_accepted_tokens += 1
                accepted_per_pos[j] += 1
                accepted_this_round += 1

                if eos_token_id is not None and draft_next == eos_token_id:
                    return generated_ids, num_drafts, num_draft_tokens, num_accepted_tokens
                if len(generated_ids) >= max_tokens:
                    break

                prev_token = draft_next
                prev_pos += 1
                target_next, target_hidden = _target_next_and_hidden(
                    target_model,
                    context_ids,
                    target_device,
                )
            else:
                context_ids.append(target_next)
                generated_ids.append(target_next)
                rejected = True
                if eos_token_id is not None and target_next == eos_token_id:
                    return generated_ids, num_drafts, num_draft_tokens, num_accepted_tokens
                break

        if len(generated_ids) >= max_tokens:
            break

        # Standard speculative step: if all draft tokens accepted, append one target token.
        if (not rejected) and accepted_this_round == draft_len:
            context_ids.append(target_next)
            generated_ids.append(target_next)
            if eos_token_id is not None and target_next == eos_token_id:
                break

    return generated_ids, num_drafts, num_draft_tokens, num_accepted_tokens


def _resolve_target_device(requested_device, tp):
    if requested_device and requested_device != "auto":
        return torch.device(requested_device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if int(tp) == 1 and n >= 2:
        # Keep vLLM on cuda:0, place HF target verifier on another GPU.
        return torch.device("cuda:1")
    return torch.device("cuda:0")


def main():
    cfg = json.loads(sys.argv[1])
    if int(cfg.get("tp", 1)) != 1 or int(cfg.get("draft_tp", 1)) != 1:
        raise ValueError("Explicit-forward probe currently supports tp=1 and draft-tp=1 only.")

    torch.manual_seed(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["target_model"],
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    target_device = _resolve_target_device(cfg.get("target_device", "auto"), cfg.get("tp", 1))
    target_dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    target_model = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        trust_remote_code=cfg.get("trust_remote_code", True),
        torch_dtype=target_dtype,
        low_cpu_mem_usage=True,
    )
    target_model.to(target_device)
    target_model.eval()

    speculative_config = {
        "model": cfg["draft_model"],
        "method": cfg["method"],
        "num_speculative_tokens": cfg["num_speculative_tokens"],
        "draft_tensor_parallel_size": cfg["draft_tp"],
    }

    max_prompt_tokens = 1
    for p in cfg["prompts"]:
        ids = tokenizer(p, return_tensors="pt")["input_ids"][0]
        max_prompt_tokens = max(max_prompt_tokens, int(ids.numel()))
    max_model_len = max(256, max_prompt_tokens + int(cfg["max_tokens"]) + 16)

    llm = None
    try:
        llm = LLM(
            model=cfg["target_model"],
            tensor_parallel_size=cfg["tp"],
            trust_remote_code=cfg.get("trust_remote_code", True),
            enforce_eager=True,
            disable_log_stats=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=float(cfg.get("vllm_gpu_memory_utilization", 0.85)),
            speculative_config=speculative_config,
        )

        speculator, spec_path = _find_speculator(llm)
        if speculator is None:
            raise RuntimeError("Could not locate internal EAGLE speculator object.")
        eagle_model = getattr(speculator, "model", None)
        if eagle_model is None:
            raise RuntimeError("Speculator found, but `.model` is missing.")
        eagle_vllm_config = getattr(speculator, "vllm_config", None)
        if eagle_vllm_config is None:
            raise RuntimeError("Speculator found, but `.vllm_config` is missing.")
        eagle_model.eval()

        accepted_per_pos = [0] * int(cfg["num_speculative_tokens"])
        total_num_drafts = 0
        total_num_draft_tokens = 0
        total_num_accepted_tokens = 0
        total_output_tokens = 0
        results = []

        with torch.inference_mode():
            for prompt in cfg["prompts"]:
                gen_ids, n_drafts, n_draft_toks, n_acc = _spec_decode_prompt(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    target_model=target_model,
                    eagle_model=eagle_model,
                    eagle_vllm_config=eagle_vllm_config,
                    num_speculative_tokens=int(cfg["num_speculative_tokens"]),
                    max_tokens=int(cfg["max_tokens"]),
                    accepted_per_pos=accepted_per_pos,
                )
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append({
                    "text": text,
                    "token_ids": [int(x) for x in gen_ids],
                })
                total_num_drafts += int(n_drafts)
                total_num_draft_tokens += int(n_draft_toks)
                total_num_accepted_tokens += int(n_acc)
                total_output_tokens += int(len(gen_ids))

        acceptance_rate = (
            float(total_num_accepted_tokens) / float(total_num_draft_tokens)
            if total_num_draft_tokens > 0 else 0.0
        )
        mean_acceptance_length = (
            1.0 + float(total_num_accepted_tokens) / float(total_num_drafts)
            if total_num_drafts > 0 else 1.0
        )
        acceptance_per_pos = [
            (float(v) / float(total_num_drafts) if total_num_drafts > 0 else 0.0)
            for v in accepted_per_pos
        ]

        with open(cfg["output_file"], "w") as f:
            json.dump({
                "results": results,
                "acceptance": {
                    "num_drafts": total_num_drafts,
                    "num_draft_tokens": total_num_draft_tokens,
                    "num_accepted_tokens": total_num_accepted_tokens,
                    "acceptance_rate": acceptance_rate,
                    "mean_acceptance_length": mean_acceptance_length,
                    "acceptance_per_pos": acceptance_per_pos,
                    "total_output_tokens": total_output_tokens,
                    "metrics_available": False,
                    "metrics_error": "manual explicit-forward mode (no llm.get_metrics)",
                    "speculator_path": spec_path,
                    "target_device": str(target_device),
                },
            }, f)
    finally:
        if llm is not None:
            try:
                llm.llm_engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
'''


def run_worker(script: str, config: dict, label: str) -> dict | None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp",
    ) as f:
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
            [sys.executable, script_path, json.dumps(config)],
            timeout=3600,
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


def main():
    parser = argparse.ArgumentParser(
        description="Ref-only vLLM EAGLE explicit-forward probe with acceptance report",
    )
    parser.add_argument("--target-model", required=True, help="Target/base model name")
    parser.add_argument("--draft-model", required=True, help="EAGLE/EAGLE3 draft model name")
    parser.add_argument(
        "--method",
        default="eagle",
        choices=["eagle", "eagle3"],
        help="Speculative method (default: eagle)",
    )
    parser.add_argument("--tp", type=int, default=1, help="Target TP (default: 1)")
    parser.add_argument("--draft-tp", type=int, default=1, help="Draft TP (default: 1)")
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=4,
        help="Number of speculative tokens (default: 4)",
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens per prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--target-device",
        default="auto",
        help=(
            "Device for HF target verifier model. "
            "Default auto: cuda:1 when tp=1 and >=2 GPUs, else cuda:0/cpu."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization for explicit mode (default: 0.85)",
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True (default: True)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code",
    )
    parser.add_argument(
        "--ref-only",
        action="store_true",
        default=True,
        help="Reference-only mode (this script only supports ref-only)",
    )
    args = parser.parse_args()

    if not args.ref_only:
        raise ValueError("This script only supports --ref-only mode.")

    print("=" * 70)
    print("  vLLM EAGLE Alignment — Explicit Forward (Reference Only)")
    print("=" * 70)
    print(f"  Target model          : {args.target_model}")
    print(f"  Draft model           : {args.draft_model}")
    print(f"  Method                : {args.method}")
    print(f"  TP / Draft TP         : {args.tp} / {args.draft_tp}")
    print(f"  Num speculative toks  : {args.num_speculative_tokens}")
    print(f"  Max tokens            : {args.max_tokens}")
    print(f"  Seed                  : {args.seed}")
    print(f"  Target device         : {args.target_device}")
    print(f"  vLLM mem util         : {args.vllm_gpu_memory_utilization}")
    print(f"  Prompts               : {len(PROMPTS)}")
    print(f"  Trust RC              : {args.trust_remote_code}")
    print("=" * 70)

    config = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "method": args.method,
        "tp": args.tp,
        "draft_tp": args.draft_tp,
        "num_speculative_tokens": args.num_speculative_tokens,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "target_device": args.target_device,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "prompts": PROMPTS,
        "trust_remote_code": args.trust_remote_code,
    }
    data = run_worker(VLLM_EAGLE_WORKER, config, "vLLM explicit target+draft run")
    if data is None:
        print("\nFAIL: worker run failed.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("  GENERATED OUTPUTS")
    print(f"{'=' * 70}")
    for i, r in enumerate(data["results"]):
        prompt_preview = PROMPTS[i][:60] + ("..." if len(PROMPTS[i]) > 60 else "")
        print(f"\n  Prompt #{i}: {prompt_preview}")
        print(f"  Tokens   : {len(r['token_ids'])}")
        print("  Output   :")
        for line in r["text"].splitlines():
            print(f"    {line}")

    acc = data["acceptance"]
    print(f"\n{'=' * 70}")
    print("  SPECULATIVE DECODING ACCEPTANCE REPORT")
    print(f"{'=' * 70}")
    print(f"  num_drafts             : {acc['num_drafts']}")
    print(f"  num_draft_tokens       : {acc['num_draft_tokens']}")
    print(f"  num_accepted_tokens    : {acc['num_accepted_tokens']}")
    print(f"  acceptance_rate        : {acc['acceptance_rate']:.4f}")
    print(f"  mean_accept_length     : {acc['mean_acceptance_length']:.4f}")
    print(f"  total_output_tokens    : {acc['total_output_tokens']}")
    print(f"  metrics_available      : {acc.get('metrics_available', False)}")
    if acc.get("metrics_error"):
        print(f"  metrics_error          : {acc.get('metrics_error')}")
    if acc.get("speculator_path"):
        print(f"  speculator_path        : {acc.get('speculator_path')}")
    if acc.get("target_device"):
        print(f"  target_device_used     : {acc.get('target_device')}")
    per_pos = acc.get("acceptance_per_pos", [])
    for i, v in enumerate(per_pos):
        print(f"  acceptance_at_token_{i}: {v:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
