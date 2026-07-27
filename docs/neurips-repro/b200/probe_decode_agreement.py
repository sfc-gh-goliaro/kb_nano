"""Do TRTLLM-gen decode and FlashInfer's wrapper disagree with each other?

vLLM's decode auto-detection is `use_trtllm = num_tokens <= 256` (see
vllm/utils/flashinfer.py:use_trtllm_attention), so above a 256-token decode batch
the reference silently switches from TRTLLM-gen to
BatchDecodeWithPagedKVCacheWrapper on the *same* HND cache. We call TRTLLM-gen at
every batch size. probe_decode_accuracy.py showed both kernels sit ~2.2e-3 from
fp32, but two kernels can each be 2e-3 from the truth and 4e-3 from each other --
and it is the kernel-to-kernel gap that flips a greedy argmax.

This measures that gap directly, on one shared HND page-16 cache.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from fastkernels.tasks.baseline.L1.flashinfer_decode import TRTLLMDecode

HQ, HKV, D = 32, 8, 128
SEQ = 1024
PAGE = 16


def reference(q, k, v, scale):
    B = q.shape[0]
    out = torch.empty(B, HQ, D, dtype=torch.float32, device=q.device)
    g = HQ // HKV
    for b in range(B):
        kb = k[b].float().repeat_interleave(g, dim=1)
        vb = v[b].float().repeat_interleave(g, dim=1)
        s = torch.einsum("hd,shd->hs", q[b].float(), kb) * scale
        out[b] = torch.einsum("hs,shd->hd", s.softmax(-1), vb)
    return out


def build_hnd(k, v):
    B = k.shape[0]
    ppr = SEQ // PAGE
    nblk = B * ppr
    bt = torch.arange(nblk, dtype=torch.int32, device=k.device).view(B, ppr)
    kc = torch.empty(nblk, HKV, PAGE, D, dtype=k.dtype, device=k.device)
    vc = torch.empty_like(kc)
    for b in range(B):
        for p in range(ppr):
            sl = slice(p * PAGE, (p + 1) * PAGE)
            kc[bt[b, p]] = k[b, sl].permute(1, 0, 2)
            vc[bt[b, p]] = v[b, sl].permute(1, 0, 2)
    return kc, vc, bt


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    scale = D ** -0.5
    trt = TRTLLMDecode(HQ, HKV, D)
    ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")

    print(f"seq_len={SEQ}, {HQ}q/{HKV}kv heads, head_dim {D}, HND page {PAGE}")
    print(f"{'batch':>6}  {'trtllm vs fp32':>22}  {'wrapper vs fp32':>22}  "
          f"{'trtllm vs wrapper':>24}")
    for B in (1, 8, 64, 256, 512, 1000):
        q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device=dev) * 0.3
        k = torch.randn(B, SEQ, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3
        v = torch.randn(B, SEQ, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3
        seqlens = torch.full((B,), SEQ, dtype=torch.int32, device=dev)
        kc, vc, bt = build_hnd(k, v)
        exp = reference(q, k, v, scale)

        a = trt(q, kc, vc, cache_seqlens=seqlens, block_table=bt,
                softmax_scale=scale, max_seq_len=SEQ).reshape(B, HQ, D).float()

        ppr = SEQ // PAGE
        indptr = torch.arange(0, (B + 1) * ppr, ppr, dtype=torch.int32, device=dev)
        indices = bt.reshape(-1)
        last = torch.full((B,), PAGE, dtype=torch.int32, device=dev)
        wrapper.plan(indptr, indices, last, HQ, HKV, D, PAGE,
                     pos_encoding_mode="NONE", q_data_type=torch.bfloat16,
                     kv_data_type=torch.bfloat16, sm_scale=scale)
        b_out = wrapper.run(q, (kc, vc)).reshape(B, HQ, D).float()
        torch.cuda.synchronize()

        def rel(x, y):
            return ((x - y).norm() / y.norm()).item()

        def mincos(x, y):
            return F.cosine_similarity(x.reshape(B, -1), y.reshape(B, -1),
                                       dim=-1).min().item()
        print(f"{B:>6}  {f'{rel(a, exp):.2e}':>22}  {f'{rel(b_out, exp):.2e}':>22}  "
              f"{f'rel {rel(a, b_out):.2e} cos {mincos(a, b_out):.6f}':>24}")
    print("\nvLLM would use trtllm only for the batch<=256 rows; we use it for all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
