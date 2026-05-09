# Independent re-audit self-review addendum

This addendum answers whether the first report was clearly logged and whether I independently checked the subagent work.

## What I checked after the first report

I re-read the report and verification log, then validated the proposed flip list against `_reaudit_final_v11.json`:

- All 27 proposed `composable -> partial` rows are currently `composable` in the final artifact.
- `edgetam` is currently `composable` and is the only counted `composable -> unsupported` flip.
- `solar_open` is currently `partial` and is the only counted `partial -> composable` flip.
- Starting counts are confirmed as 27 L4 / 238 composable / 170 partial / 12 unsupported.

I then re-opened the weaker/subagent-derived findings directly in the parent agent. The core examples re-checked were:

- `clap`: HF `ClapAudioSelfAttention` has a learned `relative_position_bias_table` and indexed additive bias. This matches the same V1-RPB rule used for `swin`/`donut_swin`, so the flip to `partial` is consistent.
- `data2vec_vision`: HF `Data2VecVisionSelfAttention` / SDPA path builds or accepts relative-position bias and passes it into logits/`attn_mask`. This matches the BEiT/RPB rule, so the flip to `partial` is consistent.
- `clvp`: HF applies partial RoPE to sliced q/k/v and concatenates passthrough channels. kb `RotaryEmbedding` rotates q/k full-head only; the flip to `partial` is consistent.
- `colmodernvbert`: HF uses `self.vlm = AutoModel.from_config(config.vlm_config)` and the wrapped `modernvbert` row is already partial. The flip to `partial` is consistent with the AutoModel rule.
- `edgetam`: HF `EdgeTamVisionConfig` defaults to `AutoConfig.from_pretrained("timm/repvit_m1.dist_in1k", ...)` / `timm_wrapper`. This supports `unsupported` under the existing transitive-timm rule.
- `sew`: HF `SEWPositionalConvEmbedding` applies `weight_norm` to Conv1d. kb has Conv1d but no general weight-norm wrapper for this family; the flip to `partial` is consistent.
- `starcoder2`: HF uses decoder `LayerNorm`, `gelu_pytorch_tanh` MLP, residual dropout, and sliding-window attention. This is not the kb Llama/RMSNorm/SwiGLU wrapper; the flip to `partial` is consistent.
- `swin2sr`: HF super-resolution heads use `nn.PixelShuffle` and nearest+conv upsampling. I found no kb PixelShuffle op; flip to `partial` is consistent.
- `ministral3`: HF applies Llama4-style attention scaling after RoPE on normal RoPE layers. kb `Llama4Attention` only applies temperature tuning on NoPE layers; flip to `partial` is consistent.
- `mistral4`: HF combines MLA, optional interleaved RoPE, and Llama4 query scaling after concatenating q_pass/q_rot. kb DeepSeek MLA does not expose the same scaling path; flip to `partial` is consistent.
- `pp_doclayout_v2`: HF has bbox relation embedding, unscaled additive `rel_2d_pos`, and CogView-style attention. This matches the additive-bias wrapper gap; flip to `partial` is consistent.
- `vaultgemma`: HF has GemmaRMSNorm-style `(1 + weight)`, `gelu_pytorch_tanh` gated MLP, attention softcap, final logit softcap, scaled embeddings, and sliding/full layer masks. kb has some softcap support in Gemma4 LM head and attention impl plumbing, but not the VaultGemma wrapper as a whole; flip to `partial` remains consistent.

## How I treated subagent findings

I did inspect the subagent outputs before writing the report. I did not blindly include every subagent recommendation.

Accepted into headline:
- Consistency-rule flips like `convbert`, `deberta`, `depth_*`, `dpt`, `upernet`, `encodec`, `jais2`, `vitdet`, `voxtral_realtime`.
- L4/borderline findings that affected scoping but not necessarily headline counts.
- `solar_open` partial-rotary correction.
- `edgetam` transitive-timm unsupported finding.

Not accepted as high-confidence headline flips:
- Many shard-trusted subagent “flip all shard-trusted” recommendations. I used them as warning signals, but only counted rows where either I or another focused pass had a concrete rule-based status defect.
- BART-style seq2seq exact-L2 parity rows. I left these as ambiguity/rationale defects, not counted flips, because the audit sometimes allows L1 decomposition.
- `bamba`, `siglip2` L4 scope, SAM3 L4 scope, `superpoint`, and `pp_doclayout_v3`. These remain explicitly ambiguous or strictness-dependent.

## Are decisions clearly logged?

Yes for the issue rows and recurring rules:
- `/tmp/independent_reaudit_report.md` logs R1-R8 with pattern, decision, and folder lists.
- It logs every counted flip and the headline math.
- It logs the denominator identity defect and source-drift note.
- It logs pure ambiguities separately.

Limitations in logging:
- The main report does not contain every line range opened by every subagent for all unchanged rows, because that would be hundreds of entries. It lists parent-agent high-impact reads and summarizes subagent batch coverage.
- For a merge-ready formal audit artifact, I would convert the subagent outputs into a machine-readable per-folder evidence JSON. The current `/tmp/independent_reaudit_verification_log.json` is a compact summary, not a 447-row evidence ledger.

## Consistency check result

The rule application is internally consistent for the counted changes:

- Learned/additive attention-bias generator -> `partial` across `beit`, `clap`, `data2vec_vision`, `deberta`, `vitdet`, and already-partial contrast folders.
- AutoBackbone / unconstrained AutoModel delegation -> `partial` or `unsupported` if the default route is a hard external dependency.
- WeightNorm -> `partial` across `encodec`, `sew`, and already-partial audio rows.
- Non-gated squared-ReLU / decoder LayerNorm mismatch -> `partial` for `jais2`, matching `nemotron`/GPT-family partials.
- Streaming Conv1d padding cache -> `partial` across `voxtral_realtime`, `vibevoice_*`, and codec contrast rows.
- Partial-RoPE -> requires runtime evidence, not config-only evidence; this flips `solar_open` back and leaves `bamba` ambiguous.

## Second-pass count conclusion

No count changes from the first report after this self-review.

Primary suggested headline remains:

- 27 L4 / 210 composable / 196 partial / 14 unsupported
- Strict: 237 / 447 = 53.02%
- Loose: 433 / 447 = 96.87%
- Unsupported: 14 / 447 = 3.13%

Alternative if `torchaudio` is treated as allowed and `higgs_audio_v2_tokenizer` is partial rather than unsupported:

- 27 L4 / 210 composable / 197 partial / 13 unsupported
- Strict: 237 / 447 = 53.02%
- Loose: 434 / 447 = 97.09%
- Unsupported: 13 / 447 = 2.91%


## Fresh spot checks after user challenge

After the user challenged whether the self-review relied on memory, I re-opened representative HF and kb-nano implementation files again. These were not memory-based:

### Spot check A: `convbert` composable -> partial

HF files re-opened:
- `/tmp/hf_transformers_pinned/src/transformers/models/convbert/modeling_convbert.py:109-254`

kb-nano files re-opened:
- `tasks/baseline/L1/conv1d.py:1-66`
- `tasks/baseline/L2/encoder_attention.py:1-151`
- searched `tasks/baseline` for `unfold`, `nn.Unfold`, and `functional.unfold`; no matches.

Re-derived conclusion:
HF `ConvBertSelfAttention` is not just Conv1d + encoder attention. It builds dynamic convolution kernels from `mixed_key_conv_attn_layer * mixed_query_layer`, applies `nn.functional.unfold`, matmuls unfolded conv windows with dynamic kernels, and concatenates that result with standard attention output. kb-nano has plain Conv1d and encoder attention but no unfold/dynamic-conv-attention wrapper. The `partial` flip is still justified.

### Spot check B: `edgetam` composable -> unsupported

HF files re-opened:
- `/tmp/hf_transformers_pinned/src/transformers/models/edgetam/modular_edgetam.py:62-231`

kb-nano checks:
- searched `tasks/baseline` for `timm_wrapper`, `TimmWrapper`, `repvit`, and `edgetam`.

Re-derived conclusion:
HF `EdgeTamVisionConfig.__post_init__` defaults `backbone_config` to `AutoConfig.from_pretrained("timm/repvit_m1.dist_in1k", model_args={...})`; if a dict is supplied, it defaults `model_type` to `timm_wrapper`. The active `EdgeTamVisionModel.forward` calls `self.backbone(pixel_values, **kwargs)`. kb-nano has a few specific timm-derived L4 loaders (`swinv2`, `siglip2`, `dinov3`, `mobilenetv4`) but no generic `timm_wrapper`/RepViT EdgeTAM support. Under the audit’s external timm rule, `unsupported` is still justified.

### Spot check C: `clap` composable -> partial

HF files re-opened:
- `/tmp/hf_transformers_pinned/src/transformers/models/clap/modeling_clap.py:346-423`

kb-nano file previously opened and re-considered:
- `tasks/baseline/L2/swinv2_window_attention.py:1-177`

Re-derived conclusion:
HF `ClapAudioSelfAttention` is copied from Swin V1 and uses a learned `relative_position_bias_table`, an indexed `relative_position_index`, and additive bias on attention scores. kb `SwinV2WindowAttention` implements cosine attention plus continuous position bias MLP, not V1 learned RPB. This is the same pattern as already-partial `swin`/`donut_swin`, so `clap -> partial` is consistent.

### Spot check D: `ministral3` and `mistral4` composable -> partial

HF files re-opened:
- `/tmp/hf_transformers_pinned/src/transformers/models/ministral3/modeling_ministral3.py:90-249`
- `/tmp/hf_transformers_pinned/src/transformers/models/mistral4/modeling_mistral4.py:205-544`

kb-nano files re-opened:
- `tasks/baseline/L2/llama4_attention.py:1-130`
- `tasks/baseline/L2/deepseek_mla_attention.py:1-228`
- `tasks/baseline/L2/attention_impl.py:400-479`

Re-derived conclusion:
HF `Ministral3Attention` applies Llama4-style attention scaling after RoPE (`query_states = query_states * get_llama_4_attn_scale(...)`). kb `Llama4Attention` only enables temperature tuning on NoPE layers (`attn_temperature_tuning and nope`) and applies it only when `nope` is true. HF `Mistral4Attention` combines MLA-style q/pass q/rot splitting, optional interleaved RoPE, then applies the same query scaling after concatenation. kb `DeepSeekMLAAttention` handles MLA/YARN/DSA-style components but has no matching post-concat Llama4 query-scale path. Both `partial` flips remain justified.

### Spot check E: `swin2sr` composable -> partial

HF files re-opened:
- `/tmp/hf_transformers_pinned/src/transformers/models/swin2sr/modeling_swin2sr.py:812-1019`

kb-nano checks:
- searched `tasks/baseline` for `PixelShuffle` / `pixel_shuffle`; no implementation found.

Re-derived conclusion:
The SwinV2 attention component is covered, but HF `Swin2SRForImageSuperResolution` includes upsampler heads using `nn.PixelShuffle` (`Upsample`, `UpsampleOneStep`, `PixelShuffleUpsampler`, `PixelShuffleAuxUpsampler`) and nearest-conv upsampling branches. Since kb-nano has no PixelShuffle op/wrapper, full-folder `composable` is not consistent.

## Transparency correction

The first self-review was fast because it was a targeted review of the existing report, verification log, and selected high-risk files, not a full re-read of all 447 folders. This fresh spot-check pass re-opened and re-derived several counted findings directly. It increases confidence in the counted examples above, but it still is not a full second full-corpus audit.
