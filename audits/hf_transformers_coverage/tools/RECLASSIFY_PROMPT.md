# Reclassify partial / unsupported folders by paper definition

The current audit marks folders `partial` or `unsupported` whenever there's no
EXACT L2 match in kb-nano. The paper's definition is looser: a folder is
**composable** if no NEW compute *primitive* (L1 op) is required — i.e.,
kb-nano has all the L1 building blocks even if the L2 wrapper doesn't exist.

For each folder in your list, determine the correct status under the paper's
definition:

- **composable**: every kernel-bearing operation in this HF folder has a
  corresponding L1 op in `/home/olu/kb_nano/tasks/baseline/L1/`. The L2
  wrapper may not exist yet but the primitives do. Examples that should
  qualify: ALiBi (= bias-add to attention scores; L1 has dense_attention.py),
  block-sparse (= mask + dense_attention), simple BART-style cross-attn
  (= linear + dense_attention + KV cache).
- **partial**: kb-nano is missing an op that's NOT a torch.nn primitive AND
  NOT a trivial composition of existing L1 ops. Example: ALiBi via
  `nn.functional.linspace` arithmetic counts as torch.nn fallback (looser).
- **unsupported**: kb-nano is missing a NEW compute primitive that genuinely
  needs implementation. Examples: FFT (fnet), LSH attention (reformer),
  RWKV-1/4 wkv (rwkv), MRA's mra2_attention (mra), xLSTM matrix-LSTM (xlstm),
  Conformer relative-position einsum (parakeet, fastspeech2_conformer).

For each folder in your shard, output:
```
folder_name: <new_status> — <one-line reason>
```

Be honest. Don't reclassify everything to composable; only reclassify those
where you can verify (by reading the HF source AND checking the kb-nano L1
listing at /home/olu/kb_nano/audits/hf_transformers_coverage/tools/kb_nano_files.txt)
that all primitives are present.

Output to: /home/olu/kb_nano/audits/hf_transformers_coverage/tools/reclassify_<shard_letter>.md

Reference: kb-nano L1 files are in `/home/olu/kb_nano/tasks/baseline/L1/`.
HF source is at `/tmp/hf_transformers_pinned/src/transformers/models/<folder>/`.
The original agent rationale (where each class is mapped) is in
`/home/olu/kb_nano/audits/hf_transformers_coverage/tools/manual_audit_shard_*.md`.
