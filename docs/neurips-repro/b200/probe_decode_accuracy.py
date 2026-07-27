"""Which Blackwell decode path is closer to the truth, and how does that scale?

Full-scale Llama-3.1 agrees with vLLM on 21-27 tokens on the TRTLLM/HND path and
205-256 on the flash_attn/NHD path, and within one run agreement degrades as batch
occupancy grows. Agreement is not accuracy, though: it says the two stacks differ,
not which one is wrong. So compare both real decode kernels against an fp32 PyTorch
reference on the same paged inputs, sweeping batch size.

Llama-3.1-8B shapes: 32 q heads, 8 kv heads, head_dim 128.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fastkernels.tasks.baseline.L1.flashinfer_decode import TRTLLMDecode
from fastkernels.tasks.baseline.L1.flash_attn_decode import FlashAttnDecode

HQ, HKV, D = 32, 8, 128
SEQ = 1024
TRTLLM_PAGE = 16     # AttnBackendConfig: Blackwell HND
FA_PAGE = 256        # AttnBackendConfig: Hopper NHD


def reference(q, k, v, scale):
    """fp32 attention, one page-free sequence per batch element. [B,HQ,D]"""
    B = q.shape[0]
    out = torch.empty(B, HQ, D, dtype=torch.float32, device=q.device)
    g = HQ // HKV
    for b in range(B):
        qb = q[b].float()                                  # [HQ, D]
        kb = k[b].float().repeat_interleave(g, dim=1)       # [SEQ, HQ, D]
        vb = v[b].float().repeat_interleave(g, dim=1)
        s = torch.einsum("hd,shd->hs", qb, kb) * scale
        out[b] = torch.einsum("hs,shd->hd", s.softmax(-1), vb)
    return out


def build_paged(k, v, page, layout):
    """Scatter contiguous [B,SEQ,HKV,D] K/V into a paged cache + block table."""
    B = k.shape[0]
    ppr = SEQ // page
    nblk = B * ppr
    bt = torch.arange(nblk, dtype=torch.int32, device=k.device).view(B, ppr)
    if layout == "HND":
        kc = torch.empty(nblk, HKV, page, D, dtype=k.dtype, device=k.device)
        vc = torch.empty_like(kc)
        for b in range(B):
            for p in range(ppr):
                sl = slice(p * page, (p + 1) * page)
                kc[bt[b, p]] = k[b, sl].permute(1, 0, 2)
                vc[bt[b, p]] = v[b, sl].permute(1, 0, 2)
    else:  # NHD
        kc = torch.empty(nblk, page, HKV, D, dtype=k.dtype, device=k.device)
        vc = torch.empty_like(kc)
        for b in range(B):
            for p in range(ppr):
                sl = slice(p * page, (p + 1) * page)
                kc[bt[b, p]] = k[b, sl]
                vc[bt[b, p]] = v[b, sl]
    return kc, vc, bt


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    scale = D ** -0.5
    trt = TRTLLMDecode(HQ, HKV, D)
    fa = FlashAttnDecode(HQ, HKV, D)
    print(f"paged decode error vs fp32 reference  (seq_len={SEQ}, "
          f"{HQ}q/{HKV}kv heads, head_dim {D})")
    print(f"{'batch':>6}  {'TRTLLM/HND page16':>28}  {'flash_attn/NHD page256':>28}")
    for B in (1, 8, 32, 128, 512):
        q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device=dev) * 0.3
        k = torch.randn(B, SEQ, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3
        v = torch.randn(B, SEQ, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3
        seqlens = torch.full((B,), SEQ, dtype=torch.int32, device=dev)
        exp = reference(q, k, v, scale)

        cells = []
        for op, page, layout in ((trt, TRTLLM_PAGE, "HND"), (fa, FA_PAGE, "NHD")):
            kc, vc, bt = build_paged(k, v, page, layout)
            got = op(q, kc, vc, cache_seqlens=seqlens, block_table=bt,
                     softmax_scale=scale, max_seq_len=SEQ)
            got = got.reshape(B, HQ, D).float()
            torch.cuda.synchronize()
            cos = F.cosine_similarity(got.reshape(B, -1), exp.reshape(B, -1),
                                      dim=-1)
            rel = ((got - exp).norm() / exp.norm()).item()
            cells.append(f"cos {cos.min().item():.6f} rel {rel:.2e}")
        print(f"{B:>6}  {cells[0]:>28}  {cells[1]:>28}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
