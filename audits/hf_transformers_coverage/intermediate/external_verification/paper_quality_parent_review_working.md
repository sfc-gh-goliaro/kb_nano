# Paper-quality re-audit parent review notes (working)

## Corrections to fresh A-C subaudit

The A-C subaudit used `unsupported` too broadly. Under the fixed methodology, a row is unsupported only when active/default inference requires hard external runtime or non-torch custom kernel. Rows that are torch-decomposable but missing kb wrappers are `partial`.

Parent-corrected A-C examples:

- `autoformer`: HF uses `torch.fft.rfft/irfft`, top-k, roll/gather autocorrelation. kb has no FFT/autocorrelation wrapper, but this is torch-decomposable. Status should be `partial`, not `unsupported`.
- `big_bird`: HF block-sparse attention uses torch/bmm/gather/random-block logic. No kb BigBird wrapper, but no hard external dependency. Status should be `partial`, not `unsupported`.
- `bigbird_pegasus`: same BigBird block-sparse + seq2seq wrapper gap. Status should be `partial`, not `unsupported`.
- `bit`: HF WeightStandardizedConv2d uses `F.batch_norm` over conv weights then `F.conv2d`. Missing kb wrapper but torch-decomposable. Status should be `partial`, not `unsupported`.
- `canine`: HF has hash-bucket embeddings and downsample/upsample character-to-molecule flow using embeddings/Conv1d/transformer layers. Missing kb row/wrapper, but torch-decomposable. Status should be `partial`, not `unsupported`.

A-C subaudit useful but not final; parent review required before counts.


## Parent review of N-S discrepancy

- `openai_privacy_filter`: parent re-opened HF `modeling_openai_privacy_filter.py:1-240`, config `configuration_openai_privacy_filter.py:1-118`, kb `L4/gpt_oss.py:1-220`, `L2/gpt_oss_moe.py:1-180`, and searched kb surface for interleaved/softcap/GPT-OSS features.
- Conclusion: `openai_privacy_filter` should be `partial` under strict rules. It is not covered by kb GPT-OSS: HF uses interleaved RoPE (`x[..., ::2]` / `x[..., 1::2]`), bidirectional sliding mask, sinks, per-Q/K scaling `head_dim**-0.25`, and unquantized/bespoke expert path, while kb GPT-OSS is OpenAI GPT-OSS MXFP4/YaRN causal model-specific.
