# L3/L4 agent run -- Claude Code vs Codex

Generated kernels and the prompts that produced them, kept because they are the
evidence behind the reported numbers: without them a result like `vision_block`
at 1.81x or `gpt_oss` at 11841x cannot be checked by anyone else.

## Layout

    prompts/<target>.md          the prompt, byte-identical for both agents
    claude-opus-4-6/<target>.py  Claude Code's candidate
    codex-gpt-5.5/<target>.py    Codex's candidate
    results-*.txt                per-target correctness and speedup

These are *artifacts*, not code the harness loads. Benchmarking copies one into
`tasks/candidate/L<n>/<target>.py`, runs, and deletes it; the candidate slot is
left empty.

## How they were produced

Prompts come from `agent.build_generation_prompt()` (unmodified), written to
`task.md` in a per-target directory. Each agent was started in that directory
and told to read it:

    cd $W && claude -p "Read task.md and carry out its instructions. Write the
      answer to out.py ..." --model claude-opus-4-6 --effort high
      --allowedTools Read,Write --permission-mode acceptEdits

    cd $W && codex exec --skip-git-repo-check -m gpt-5.5
      -c model_reasoning_effort="high" -s workspace-write
      "Read task.md and carry out its instructions. Write the answer to out.py ..."

The prompt is staged on disk rather than passed on the command line because a
prompt with the baseline source inlined aborts the CLI stream.

Single-shot: neither agent could execute anything (read/write only), so neither
saw a compile error or a test result, and there was no retry.

## Measurement

    python -m fastkernels.bench.kernels --target <t> --num-warmup 10 --num-runs 30

Per scenario: N warmup forwards, then N timed forwards with a CUDA
synchronize after each, median taken; speedup = baseline_ms / candidate_ms.
Per target: mean over its scenarios. A target counts correct only if *every*
scenario passes, so an incorrect candidate contributes no speedup.

`oasis_rollout` used --num-warmup 1 --num-runs 2 for both agents: it is a
per-frame x per-DDIM-step nested loop and 30 runs does not finish a single
scenario.

Warmup matters more than it looks. At the original --num-warmup 1 --num-runs 3,
Claude's `llama_decoder` measured 1.20x; at 10/30 it is 0.98x. That candidate is
bit-identical to the baseline (it calls torch.ops._C.rms_norm directly instead
of through RMSNorm.forward), so ~1.0x is the truth.

## Caveats

- Run against the patched harness on this branch, not stock kb_nano. Most L3/L4
  targets cannot be constructed without it.
- `qwen3_vl` is Blocked for both agents: 235B, declared tp=4 in
  benchmark_scenarios/small/config.yaml, does not fit one 143 GB card.
- Codex's `gpt_oss` reports 11841x. It is a stub that skips the work; the
  correctness gate catches it (0/5). A speedup that large should always be
  treated as a candidate that does nothing until proven otherwise.
- Correctness compares return values and input tensors, not the paged KV cache.
  A decoder layer that skips its cache write would return a bit-identical output
  during a single prefill and score correct. Spot-checked four decoder
  candidates: all wrote the cache, differing only at rounding level (4-6e-3
  against a bf16 epsilon of 3.9e-3).
