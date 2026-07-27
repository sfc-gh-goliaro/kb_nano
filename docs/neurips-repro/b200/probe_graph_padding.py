"""Is the captured-graph decode path's residual divergence caused by batch padding?

State after fixing the block-table stride bug (Llama-3.1, 1000 prompts, matched tokens):

    fully eager (--enforce-eager)         475.8 / 510.6 / 887.8     <- above H200's 408.5
    default (graph <=512, eager above)    208.9 / 243.0 / 259.2
    all-graph (max_num_seqs=512)          209.0 / 243.1 / 259.1

So the captured-graph path is now the limiting one. Its one remaining structural difference
from the eager path is that it pads the decode batch up to a captured bucket, filling the
padded rows with slot_mapping=-1 and context_lens=0, whereas the eager path runs the exact
batch. probe_batch_sensitivity.py already showed TRTLLM-gen's output for a given row depends
on the batch size, so padding is a plausible cause.

This tests it directly: run the SAME decode batch through the graph path and the eager path,
once with n exactly equal to a captured bucket (no padding) and once with n just below one
(maximum padding). If padding is the cause, the no-padding case agrees and the padded case
does not.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_LEN = 400


def main() -> int:
    from fastkernels.infra.engine import LlamaEngine, SamplingParams

    engine = LlamaEngine(MODEL, max_model_len=2048, enforce_eager=False)
    mr = engine.model_runner
    buckets = mr.graph_bs_list
    print(f"captured buckets, last 4: {buckets[-4:]}")

    # The scheduler never lands on a chosen n exactly, so capture whatever decode
    # batches actually occur and compare the largest graph-eligible ones.
    captured = {}
    orig = type(mr).run_decode_greedy_fast_async

    def spy(self, decode_data):
        n = decode_data[0]
        if n not in captured:
            captured[n] = tuple(
                x.copy() if isinstance(x, np.ndarray) else x for x in decode_data)
        return orig(self, decode_data)

    type(mr).run_decode_greedy_fast_async = spy
    batch = 600
    prompts = [[1000 + (i * 7 + j) % 20000 for j in range(PROMPT_LEN)]
               for i in range(batch)]
    engine.generate(prompts,
                    [SamplingParams(temperature=0.0, max_tokens=6,
                                    ignore_eos=True)] * batch,
                    use_tqdm=False)
    type(mr).run_decode_greedy_fast_async = orig

    ceiling = mr.graph_bs_list[-1]
    eligible = sorted(n for n in captured if n <= ceiling)
    picks = eligible[-3:] + sorted(n for n in captured if n > ceiling)[:1]
    print(f"\nobserved decode batch sizes: min {min(captured)} max {max(captured)} "
          f"({len(captured)} distinct); graph ceiling {ceiling}")
    for n in picks:
        d = captured[n]
        bucket = mr._graph_bs_for_n[n] if n <= ceiling else None
        with torch.inference_mode():
            tok_eager = mr._run_decode_greedy_eager(*d).clone()
            if bucket is not None and bucket in mr.graphs:
                mr._run_graph_from_numpy(*d)
                tok_graph = mr.graph_vars["lm_max_idxs"][:n].clone()
            else:
                tok_graph = None
        torch.cuda.synchronize()
        if tok_graph is None:
            print(f"  n={n:>4}  above the ceiling -- eager only, no graph to compare")
            continue
        agree = int((tok_graph == tok_eager).sum())
        print(f"  n={n:>4} bucket={bucket:>4} padding={bucket - n:>3} rows -> "
              f"tokens agreeing graph-vs-eager: {agree}/{n}")
    print("\nIf the no-padding row agrees and the padded row does not, batch padding is\n"
          "the remaining graph-path divergence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
