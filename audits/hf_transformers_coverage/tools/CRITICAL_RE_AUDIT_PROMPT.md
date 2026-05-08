# Re-audit critical folders (partial / unsupported)

The current audit marks 28 folders as `partial` or `unsupported`. The paper
claims only 4 are truly unsupported (mra, reformer, rwkv-v4, xlstm) and ~3
are partial (torch.nn fallback). Verify each folder by reading actual HF
source and checking against kb-nano's L1 inventory.

For each folder, output exactly one line:
```
<folder>: <composable|partial|unsupported> — <one-line reason>
```

Definitions:
- **composable**: every compute op in this HF folder has an existing kb-nano
  L1 file (no new primitive needed). The L2 wrapper may not exist; that's OK.
- **partial**: needs ONE simple torch.nn fallback (e.g. `gather`, `bmm`,
  `index_select`, `clamp`, simple bias-add) — no new compute kernel.
- **unsupported**: needs a GENUINELY NEW compute primitive (FFT, LSH,
  custom CUDA, novel SSM, novel attention scan, etc.).

To decide, read:
- HF source: `/tmp/hf_transformers_pinned/src/transformers/models/<folder>/modeling_<folder>.py`
- kb-nano L1 listing: `ls /home/olu/kb_nano/tasks/baseline/L1/`
- Prior agent notes: `/home/olu/kb_nano/audits/hf_transformers_coverage/tools/manual_audit_shard_*.md`

Be honest. The strictest cases (mra, reformer, rwkv, xlstm, fnet, yoso,
nystromformer) likely stay unsupported. Many others (Conformer rel-pos,
DAC/HiFi-GAN with Snake1d, block-local attention, two-stream attention)
can be composed from L1 + simple gather/bmm — those are partial or
composable, not unsupported.

Snake1d activation = `x + (1/alpha) * sin(alpha*x)^2` — pure elementwise,
composable from existing primitives (no new kernel needed).
