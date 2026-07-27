"""Which difference between our graph and eager decode paths changes the numbers?

Established by measurement on B200, Llama-3.1-8B, 1000 prompts, same prompt set:

    all decode steps in a captured CUDA graph (max_num_seqs=512)  -> 209 / 243 / 259 matched
    all decode steps eager        (--enforce-eager)               ->  30 /  24 /  27 matched
    default (graph <=512, eager above; B200 reaches ~750 in-flight) -> 27 / 21 / 21 matched

So the eager decode path is the one that disagrees with vLLM, and B200 only falls into it
because its larger HBM admits more concurrent sequences than the 512-entry graph ceiling
(max_capture_limit in infra/engine.py) -- on H200 the KV cache cannot hold that many, which
is why H200 never showed this.

The two paths differ in exactly one attention-visible argument:

    graph capture : set_context(..., max_context_len=self.max_model_len)   # constant, baked in
    eager         : set_context(..., max_context_len=int(cl_np.max()))     # true batch max

``max_context_len`` reaches TRTLLMDecode as ``max_seq_len``
(tasks/baseline/L2/attention_impl.py: ``max_seq_len=max_ctx``), a kernel parameter of
trtllm_batch_decode_with_kv_cache. This runs one real decode batch through the model under
each convention and compares, holding everything else fixed.
"""
from __future__ import annotations

import numpy as np
import torch

from fastkernels.infra.context import get_context, reset_context, set_context

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_LEN = 600          # long enough that pages matter
BATCH = 64


def main() -> int:
    from fastkernels.infra.engine import LlamaEngine

    engine = LlamaEngine(MODEL, max_model_len=2048, enforce_eager=False)
    mr = engine.model_runner
    print(f"max_model_len={mr.max_model_len} graph_bs_list[-1]={mr.graph_bs_list[-1]} "
          f"max_num_seqs={mr.max_num_seqs}")

    captured = {}
    orig = type(mr).run_decode_greedy_fast_async

    def spy(self, decode_data):
        if "d" not in captured and decode_data[0] >= 8:
            captured["d"] = tuple(
                x.copy() if isinstance(x, np.ndarray) else x for x in decode_data)
        return orig(self, decode_data)

    type(mr).run_decode_greedy_fast_async = spy
    from fastkernels.infra.engine import SamplingParams
    prompts = [[1000 + (i * 7 + j) % 20000 for j in range(PROMPT_LEN)]
               for i in range(BATCH)]
    engine.generate(prompts,
                    [SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)] * BATCH,
                    use_tqdm=False)
    type(mr).run_decode_greedy_fast_async = orig

    if "d" not in captured:
        print("FAIL: never captured a decode batch")
        return 1
    n, ids_np, pos_np, sm_np, cl_np, bt_np = captured["d"]
    print(f"captured decode batch: n={n} context_lens min/max={cl_np.min()}/{cl_np.max()}")

    dev = "cuda"
    ids = torch.from_numpy(ids_np).to(dev)
    pos = torch.from_numpy(pos_np).to(dev)
    sm = torch.from_numpy(sm_np).to(dev)
    cl = torch.from_numpy(cl_np).to(dev)
    bt = torch.from_numpy(bt_np).to(dev)
    req_id = getattr(mr, "_decode_req_id_buf", None)
    req_id = req_id[:n] if req_id is not None else None

    def run(max_ctx, label):
        set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt,
                    max_context_len=max_ctx, req_id_per_token=req_id)
        with torch.inference_mode():
            hidden = mr.model(ids, pos)
            lm = mr.model.lm_head
            logits = lm.linear_op(hidden, lm.embedding_op.emb.weight).float()
        reset_context()
        torch.cuda.synchronize()
        print(f"  {label:<34} max_seq_len={max_ctx}")
        return hidden.float(), logits

    print("\nsame decode batch, only max_context_len differs:")
    # Careful: the KV cache is written by the attention op during these calls, so run the
    # true-max convention first and re-run it last as a self-consistency check.
    h_true, l_true = run(int(cl_np.max()), "eager convention (true batch max)")
    h_cap, l_cap = run(mr.max_model_len, "graph convention (max_model_len)")
    h_true2, l_true2 = run(int(cl_np.max()), "eager convention again (repeat)")

    def cmp(a, b, label):
        rel = ((a - b).norm() / b.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1), dim=-1).item()
        tok_a, tok_b = a.argmax(-1), b.argmax(-1)
        same = int((tok_a == tok_b).sum()) if a.dim() == 2 else -1
        print(f"  {label:<40} rel {rel:.3e}  cos {cos:.8f}"
              + (f"  tokens agreeing {same}/{a.shape[0]}" if same >= 0 else ""))

    print("\nhidden states:")
    cmp(h_true, h_cap, "eager-vs-graph convention")
    cmp(h_true, h_true2, "eager repeated (determinism control)")
    print("\nlogits:")
    cmp(l_true, l_cap, "eager-vs-graph convention")
    cmp(l_true, l_true2, "eager repeated (determinism control)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
