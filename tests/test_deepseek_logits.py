#!/usr/bin/env python3
"""Diagnostic test for DeepSeek V3.2 all-zero output tokens.

Loads the model with TP=8 and checks each stage of the pipeline:
1. Weight loading: are lm_head weights non-zero?
2. Embedding: does embed_tokens produce non-zero hidden states?
3. Single layer forward: does one decoder layer produce non-zero output?
4. Full model forward: does the full model produce non-zero hidden states?
5. Logit computation: does compute_logits produce non-zero logits?
6. Sampling: does argmax produce non-zero token IDs?

Usage:
    torchrun --nproc_per_node=8 tests/test_deepseek_logits.py
"""
import os
import sys
import torch
import torch.distributed as dist

# Ensure kb_nano is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_package_dir = os.path.dirname(_this_dir)
_project_root = os.path.dirname(_package_dir)
sys.path.insert(0, _project_root)


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    from kb_nano.infra.tp import _tp_size, _tp_rank
    assert _tp_size() == world_size

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("=" * 60)
        print("DeepSeek V3.2 Logits Diagnostic Test")
        print("=" * 60)

    # --- Load model ---
    from kb_nano.infra.weight_loader import load_model
    model, config = load_model("deepseek-ai/DeepSeek-V3.2", device, torch.bfloat16)
    model.eval()

    if rank == 0:
        print("\n[1] Weight checks:")

    # Check lm_head weight
    lm_w = model.lm_head.weight.data
    if rank == 0:
        print(f"  lm_head.weight: shape={lm_w.shape}, dtype={lm_w.dtype}, "
              f"device={lm_w.device}, norm={lm_w.float().norm():.2f}, "
              f"any_nonzero={lm_w.any().item()}")

    # Check embed_tokens weight
    emb_w = model.model.embed_tokens.weight.data
    if rank == 0:
        print(f"  embed_tokens.weight: shape={emb_w.shape}, dtype={emb_w.dtype}, "
              f"device={emb_w.device}, norm={emb_w.float().norm():.2f}, "
              f"any_nonzero={emb_w.any().item()}")

    # Check a decoder layer weight
    layer0 = model.model.layers[0]
    attn_w = layer0.self_attn.fused_qkv_a_proj.weight.data
    if rank == 0:
        print(f"  layer0.fused_qkv_a_proj.weight: shape={attn_w.shape}, "
              f"dtype={attn_w.dtype}, norm={attn_w.float().norm():.2f}")

    # Check norm weight
    norm_w = model.model.norm.weight.data
    if rank == 0:
        print(f"  final_norm.weight: shape={norm_w.shape}, "
              f"norm={norm_w.float().norm():.2f}")

    # --- Forward pass ---
    if rank == 0:
        print("\n[2] Forward pass (4 tokens, no KV cache):")

    from kb_nano.infra.context import set_context, reset_context

    input_ids = torch.tensor([1, 100, 200, 300], dtype=torch.int64, device=device)
    positions = torch.arange(4, dtype=torch.int64, device=device)

    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, 4], dtype=torch.int32, device=device),
        max_seqlen_q=4, max_seqlen_k=4,
        slot_mapping=torch.full((4,), -1, dtype=torch.int64, device=device),
    )

    with torch.inference_mode():
        # Embedding
        emb = model.model.embed_tokens(input_ids)
        if rank == 0:
            print(f"  embed output: shape={emb.shape}, "
                  f"norm={emb.float().norm():.2f}")

        # Layer-by-layer forward
        hidden = emb
        residual = None
        for i, layer in enumerate(model.model.layers):
            hidden, residual = layer(positions, hidden, residual)
            h_norm = hidden.float().norm().item()
            r_norm = residual.float().norm().item() if residual is not None else 0
            r_has_inf = residual.isinf().any().item() if residual is not None else False
            r_has_nan = residual.isnan().any().item() if residual is not None else False
            if rank == 0:
                flag = ""
                if r_has_inf:
                    flag = " <<< INF!"
                if r_has_nan:
                    flag = " <<< NAN!"
                if h_norm == 0:
                    flag = " <<< ZERO!"
                h_max = hidden.abs().max().item()
                r_max = residual.abs().max().item() if residual is not None else 0
                print(f"  layer {i:2d}: h_norm={h_norm:.4f} h_max={h_max:.2f}  "
                      f"r_norm={r_norm:.4f} r_max={r_max:.2f}{flag}")
            if r_has_inf or r_has_nan:
                break

        # Final norm
        hidden, _ = model.model.norm(hidden, residual)
        if rank == 0:
            print(f"  after final norm: norm={hidden.float().norm():.4f}, "
                  f"any_nonzero={(hidden != 0).any().item()}")

        # LM head projection (local, per-rank)
        lm_head = model.lm_head
        local_logits = lm_head.linear_op(hidden[-1:], lm_head.weight)
        top5 = local_logits[0].topk(5)
        if rank == 0:
            print(f"  local_logits (rank 0): min={local_logits.min():.2f}, max={local_logits.max():.2f}")
            print(f"  top5: ids={top5.indices.tolist()}, "
                  f"vals=[{', '.join(f'{v:.2f}' for v in top5.values.tolist())}]")

    reset_context()

    if rank == 0:
        print("\n[3] Summary:")
        if hidden.float().norm() == 0:
            print("  FAIL: Model output is all zeros!")
        else:
            print("  PASS: Model produces non-zero output")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
