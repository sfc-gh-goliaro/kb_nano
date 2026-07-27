"""Check TRTLLMMLADecode against a plain PyTorch MLA decode.

FlashMLA's dense decode does not run on Blackwell, so there is no same-kernel
reference to diff against; compute the absorbed-MLA decode directly instead.
In absorbed form the query already lives in latent+RoPE space, so

    scores = q[576] . kv[576] * scale ;  out = softmax(scores) . kv[:, :512]
"""
from __future__ import annotations

import torch

from fastkernels.tasks.baseline.L1.flashinfer_mla_decode import TRTLLMMLADecode

KV_LORA, QK_ROPE, QK_NOPE = 512, 64, 128
DIM = KV_LORA + QK_ROPE
PAGE = 64


def reference(q, kv_cache, block_table, seq_lens, scale):
    B, H, _ = q.shape
    out = torch.empty(B, H, KV_LORA, dtype=torch.float32, device=q.device)
    for b in range(B):
        n = int(seq_lens[b])
        pages = block_table[b, : (n + PAGE - 1) // PAGE].tolist()
        kv = torch.cat([kv_cache[p] for p in pages], dim=0)[:n].float()  # [n, 576]
        scores = (q[b].float() @ kv.T) * scale                          # [H, n]
        out[b] = torch.softmax(scores, dim=-1) @ kv[:, :KV_LORA]
    return out


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    B, H = 8, 16
    seq_lens = torch.tensor([37, 64, 65, 200, 511, 512, 513, 1000],
                            dtype=torch.int32, device=dev)
    max_len = int(seq_lens.max())
    # Deliberately odd page count: FlashInfer requires block_num % (128/page)
    # == 0, and Kimi-Linear hit exactly this with 23 pages at page_size 64.
    pages_per_seq = (max_len + PAGE - 1) // PAGE
    if pages_per_seq % 2 == 0:
        pages_per_seq += 1
    nblocks = B * pages_per_seq + 4

    kv_cache = (torch.randn(nblocks, PAGE, DIM, dtype=torch.bfloat16, device=dev) * 0.1)
    block_table = torch.arange(B * pages_per_seq, dtype=torch.int32,
                               device=dev).view(B, pages_per_seq)
    q = torch.randn(B, H, DIM, dtype=torch.bfloat16, device=dev) * 0.1
    scale = (QK_NOPE + QK_ROPE) ** -0.5

    op = TRTLLMMLADecode(QK_NOPE, KV_LORA, QK_ROPE)
    got = op(q, kv_cache, block_table, seq_lens,
             max_seq_len=max_len, softmax_scale=scale).float()
    torch.cuda.synchronize()
    exp = reference(q, kv_cache, block_table, seq_lens, scale)

    cos = torch.nn.functional.cosine_similarity(
        got.reshape(B, -1), exp.reshape(B, -1), dim=-1)
    err = (got - exp).abs().max().item()
    print(f"shape {tuple(got.shape)} expected {tuple(exp.shape)}")
    print(f"per-seq cosine: {[round(c, 6) for c in cos.tolist()]}")
    print(f"max abs err {err:.4g}  mean|exp| {exp.abs().mean().item():.4g}")
    ok = bool(cos.min() > 0.999) and err < 5e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
