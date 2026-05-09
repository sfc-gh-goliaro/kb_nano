# Independent re-audit findings

## §0: Denominator and artifact verification

Final row-count artifacts currently agree with each other but disagree with the prose docs.

- `/tmp/hf_transformers_pinned/src/transformers/models` has 465 model directories and 448 `modeling_*.py` files excluding `*_old.py`.
- `auto/modeling_auto.py` is an AutoModel registry, not a model row, so the effective modeling-file denominator is 447.
- `_reaudit_final_v11.json`, `audit_evidence.csv`, and row comments in `hf_coverage_rows.tex` agree on 447 rows with counts: 27 `kb_nano_l4`, 238 `composable`, 170 `partial`, 12 `unsupported`.
- `README.md`, `CAVEATS_AND_METHODOLOGY.md`, `NUMBER_DRIFT_RECONCILIATION.md`, and `paper_archive/README.md` state 27 / 237 / 171 / 12. That prose is stale relative to the machine-readable/rendered final artifacts.
- Current GitHub `main` models directory has 467 dirs, two more than the pinned tree: `hyperclovax` and `rf_detr`. I kept these as source drift, not part of the final 447 denominator.

### Denominator identity defect
The 447 count is numerically right but the row identities are wrong.

- The pinned `donut/` folder contains only `modeling_donut_swin.py`; there is no `modeling_donut.py`.
- The final audit has both `donut` (`composable`) and `donut_swin` (`partial`) rows, both sourced to `modeling_donut_swin.py` in the shard/TeX.
- The pinned tree also contains `higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py`, a real modeling file omitted from the 447 audit rows.
- Suggested fix: remove/reconcile the duplicate `donut` composable row and add `higgs_audio_v2_tokenizer`.

`higgs_audio_v2_tokenizer` source read: `modeling_higgs_audio_v2_tokenizer.py:1-648`, `configuration_higgs_audio_v2_tokenizer.py:1-168`. It imports `torchaudio`, requires the backend, constructs DAC and HuBERT via `AutoModel.from_config`, applies weight norm to acoustic convs, uses residual vector quantization/codebooks, and uses `torchaudio.functional.resample`. I classify it as `unsupported` under the external-runtime-dependency rule; if `torchaudio` is allowed as a torch-family fallback, it is still at least `partial`.

## §1: Ambiguity-resolution rules adopted

### R1: Learned/additive attention-bias generators
- Pattern: learned RPB / ALiBi / decomposed rel-pos / disentangled relative attention generated inside the module, then added to attention scores or passed as `attn_mask`.
- Decision: `partial` unless kb-nano has a matching L1/L2 wrapper for that bias generator and placement. Generic `DenseAttention(attn_mask=...)` is not enough by itself.
- Applies to: `beit`, `bloom`, `clap`, `data2vec_vision`, `deberta`, `deberta_v2`, `donut_swin`, `maskformer_swin`, `mpnet`, `sam`, `sam_hq`, `swin`, `vitdet`, `got_ocr2`, `layoutlmv3`.

### R2: AutoBackbone / arbitrary `AutoModel.from_config` routing
- Pattern: folder-level model delegates a load-bearing submodel to `load_backbone()` or unconstrained `AutoModel.from_config`.
- Decision: `partial` unless the row is explicitly scoped to a concrete child config already covered by kb-nano L4/L2. Escalate to `unsupported` if the default child is `timm`, `detectron2`, `natten`, `xlstm`, etc.
- Applies to: `conditional_detr`, `dab_detr`, `detr`, `depth_anything`, `depth_pro`, `dpt`, `mask2former`, `mm_grounding_dino`, `oneformer`, `omdet_turbo`, `tvp`, `upernet`, `colmodernvbert`, `xcodec`, `vibevoice_asr`, `vibevoice_acoustic_tokenizer`, `higgs_audio_v2_tokenizer`.

### R3: `weight_norm` parametrization
- Pattern: `nn.utils.weight_norm` or `torch.nn.utils.parametrizations.weight_norm` in the active model path.
- Decision: `partial`; kb-nano has Conv1d/ConvTranspose1d but no weight-norm wrapper or loader-folding invariant.
- Applies to: `dac`, `encodec`, `hubert`, `mimi`, `sew`, `unispeech`, `unispeech_sat`, `univnet`, `vits`, `wav2vec2`, `wavlm`, `xcodec` conversion/acoustic path, `higgs_audio_v2_tokenizer`.

### R4: Non-gated squared-ReLU MLP / decoder LayerNorm variants
- Pattern: `down(relu2(up(x)))`, decoder `LayerNorm`, or LayerNorm1P used where kb-nano wrappers assume RMSNorm + SwiGLU.
- Decision: `partial` unless a matching L2 decoder/MLP wrapper exists. `L1/squared_relu.py` alone is not enough for L2 composability.
- Applies to: `arcee`, `jais2`, `nemotron`, plus GPT-NeoX/GPT-J/CodeGen-style LayerNorm + parallel residual families.

### R5: Decomposed 2D relative-position attention
- Pattern: SAM/VitDet/MViT-style `rel_pos_h` + `rel_pos_w`, interpolation, `einsum`, then add to attention logits.
- Decision: `partial`. kb-nano has SAM3 RoPE attention, not this decomposed rel-pos path.
- Applies to: `sam`, `sam_hq`, `got_ocr2`, `vitdet`.

### R6: Streaming Conv1d padding cache
- Pattern: explicit mutable padding cache layered around Conv1d/ConvTranspose1d for streaming chunks.
- Decision: `partial`. kb-nano has plain Conv1d and FLA-style causal conv, not the HF cache wrapper semantics.
- Applies to: `voxtral_realtime`, `kyutai_speech_to_text`, `mimi`, `vibevoice_acoustic_tokenizer`, `vibevoice_asr`.

### R7: Partial-RoPE evidence
- Pattern: `partial_rotary_factor < 1.0`.
- Decision: mark `partial` only when the active/default runtime rotates fewer than `head_dim` channels via q/pass slicing or identity-tail behavior not covered by an existing wrapper. Config-only evidence is insufficient if runtime ignores it.
- Confirmed partial: `glm4v_moe`, `laguna`, `musicflamingo`, `recurrent_gemma`.
- Not confirmed: `bamba` (config says 0.5 but runtime appears to build full-head cos/sin), `solar_open` (setdefault order leaves `partial_rotary_factor=1.0`; runtime lacks q/pass split). Suggested: flip `solar_open` back to composable; mark `bamba` ambiguous unless another path proves partial runtime.

### R8: External runtime libraries in modeling forward
- Pattern: hard runtime dependency on non-torch external libraries from model code.
- Decision: `unsupported` unless the folder has a torch fallback and the external dependency is not active in the default path.
- Applies to: `dinat` (`natten`), `layoutlmv2` (`detectron2`), `timm_backbone`/`timm_wrapper` and transitive `fast_vlm`/`edgetam` (`timm`), `xlstm`, `mra`/`rwkv`/`yoso` (`kernels-community`), `higgs_audio_v2_tokenizer` (`torchaudio`, under this report's strict rule).

## §2: Per-folder findings (issues only)

### Denominator / row identity
- `donut`: current `composable`, suggested remove/reconcile. There is no `donut/modeling_donut.py` in the pinned tree; the row duplicates `donut_swin` source.
- `higgs_audio_v2_tokenizer`: missing row, suggested `unsupported` (or at least `partial` if torchaudio is allowed). Source uses torchaudio, DAC/HubERT AutoModel children, weight norm, RVQ/codebook encode/decode.

### High-confidence composable -> partial/unsupported flips
- `beit`: learned relative-position-bias table and interpolation; no matching kb wrapper. Suggested `partial`.
- `clap`: audio side copies Swin V1 learned RPB attention; kb `swinv2_window_attention.py` is V2 cosine + CPB. Suggested `partial`.
- `data2vec_vision`: BEiT-derived learned relative-position bias path. Suggested `partial`.
- `clvp`: partial RoPE over q/k/v slices; kb RoPE rotates full q/k head. Suggested `partial`.
- `convbert`: dynamic separable Conv1d attention with `nn.functional.unfold`; no kb unfold/dynamic conv-attention wrapper. Suggested `partial`.
- `deberta`: c2p/p2c disentangled relative attention with gather/talking-head options, same family as `deberta_v2`. Suggested `partial`.
- `depth_anything`: `load_backbone(config)`. Suggested `partial`.
- `depth_pro`: `AutoModel.from_config` for encoders plus `F.unfold` patch logic. Suggested `partial`.
- `dpt`: has AutoBackbone path in hybrid embeddings / depth-estimation wrapper. Suggested `partial` under folder-wide API coverage.
- `upernet`: `load_backbone(config)`. Suggested `partial`.
- `encodec`: weight-normalized Conv1d/ConvTranspose1d. Suggested `partial`.
- `sew`: weight-normalized positional Conv1d. Suggested `partial`.
- `jais2`: `Jais2MLP(NemotronMLP)` non-gated `relu2` plus decoder LayerNorm. Suggested `partial`.
- `vitdet`: decomposed MViT-style rel-pos attention. Suggested `partial`.
- `voxtral_realtime`: streaming Conv1d padding cache and all-layer realtime semantics. Suggested `partial`.
- `colmodernvbert`: wraps `modernvbert`, already partial. Suggested `partial`.
- `edgetam`: default RepViT/timm-wrapper vision path. Suggested `unsupported`.
- `gemma4_assistant`: shared-KV masking, bidirectional/sliding assisted masks, top-k masked embedding/scatter not covered by `L4/gemma4.py`. Suggested `partial`.
- `lighton_ocr`: default vision is `pixtral`, already partial; row only covers projector. Suggested `partial`.
- `xcodec`: DAC/semantic AutoModel children, RVQ/codebook, codec orchestration; no kb VQ/codebook wrapper. Suggested `partial`.
- `vibevoice_acoustic_tokenizer`: streaming Conv1d/ConvTranspose1d cache and custom ConvNeXt1d tokenizer. Suggested `partial`.
- `vibevoice_asr`: custom acoustic/semantic tokenizers, streaming conv cache, projector and stochastic acoustic sampling. Suggested `partial`.
- `vaultgemma`: Gemma2/VaultGemma softcapping, GeGLU/tanh-GELU gated MLP, scaled embeddings; not Gemma4/Llama MLP. Suggested `partial`.
- `starcoder2`: LayerNorm decoder, non-SwiGLU MLP, residual dropout and sliding-window semantics not matching kb Llama path. Suggested `partial`.
- `swin2sr`: SwinV2 attention is covered, but super-resolution model uses PixelShuffle/upsampling path absent from kb-nano. Suggested `partial`.
- `ministral3`: Llama4-style query temperature scaling after RoPE not matched by kb Llama4/attention path as currently exposed. Suggested `partial`.
- `mistral4`: MLA + interleaved RoPE + Llama4 query scaling; kb DeepSeek MLA is close but lacks the same scaling path. Suggested `partial`.
- `pp_doclayout_v2`: bbox relation embedding + additive 2D rel-pos/CogView-style attention. Suggested `partial`.

### Partial -> composable / ambiguity flips
- `solar_open`: suggested `partial -> composable`. Current source sets `partial_rotary_factor=1.0` first; later `setdefault(0.5)` is no-op, and runtime has no q/pass split.
- `bamba`: mark `needs human judgment`. Config says `partial_rotary_factor=0.5`, but runtime rotary path appears to build full-head cos/sin; do not count the demotion as verified without another source path.

### Rationale-only issues / non-flips
- `esmfold`: stays `partial`, but rationale should not say kb-nano lacks triangle attention. kb has AF3 triangle attention/multiplication, but HF ESMFold differs in bias defaults and chunked execution.
- `gpt_neox`, `gptj`, `codegen`: stay `partial`; prior fix is confirmed. Key gaps are parallel residual, LayerNorm, fc+GELU+fc MLP, partial/interleaved RoPE.
- `paddleocr_vl`: stays `composable`; prior wiring-class fix is confirmed. Mapping should add `vision_pos_embed_interpolate.py`, and note some vision attention is L1-decomposed rather than exact `vision_attention.py` parity.
- `falcon_mamba`: keep `composable`, but mapping should point to the Mamba variant with extra RMS on B/C/dt; do not promote to L4/mamba.
- `qwen2_5_vl`: stays `composable`, not L4. `qwen25_vl_encoder.py` is a HunyuanVideo text-encoder slice; `L4/qwen2_vl.py` is not a full Qwen2.5-VL promotion.
- `pp_ocrv5_*`, `pp_doclayout_v3`, `minimax_m2`, `mgp_str`, BART-style seq2seq rows: rationale should be tightened; several are composable only under L1-decomposition leniency, not exact L2 wrapper parity.
- `siglip2`: L4 file covers the NaFlex vision encoder, while HF `siglip2` also has a text/full contrastive model. L4 promotion should be scoped as vision-encoder L4 or downgraded to strict-composable for full folder; strict count unchanged if moved L4 -> composable.
- `sam3`/`sam3_tracker`/`sam3_video`: L4 implementation references original SAM3 code and HF model surfaces differ. I mark these as L4-scope ambiguities, not counted as flips in the headline below.

## §3: Cross-folder consistency findings

- V1 learned RPB was inconsistent: `swin`, `maskformer_swin`, `donut_swin` partial, but `beit`, `clap`, `data2vec_vision` composable. I apply R1 and suggest flipping those composable rows.
- AutoBackbone was inconsistent: several detector/segmentation folders partial, but depth/segmentation rows `depth_anything`, `depth_pro`, `dpt`, `upernet` composable. I apply R2 and suggest flipping them.
- Weight norm was inconsistent: `mimi`, `vits`, `univnet` partial, but `encodec` and `sew` composable. I apply R3 and suggest flipping them.
- Non-gated squared-ReLU MLP was inconsistent: `nemotron` partial but `jais2` composable. I apply R4 and flip `jais2`.
- Decomposed rel-pos was inconsistent: `sam`, `sam_hq`, `got_ocr2` partial but `vitdet` composable. I apply R5 and flip `vitdet`.
- Streaming Conv1d cache was inconsistent: `kyutai_speech_to_text` partial, but `voxtral_realtime`, `vibevoice_*` rows composable. I apply R6 and flip the composable rows.
- Partial-RoPE demotions were over-applied in at least `solar_open`, and possibly `bamba`; config-only evidence is not enough.

## §4: Aggregated counts

High-confidence issue totals:

- Denominator/row-identity issue: 1 duplicate/missing-row pair (`donut` / `higgs_audio_v2_tokenizer`).
- High-confidence composable -> partial flips: 27.
- High-confidence composable -> unsupported flips: 1 (`edgetam`).
- High-confidence partial -> composable flips: 1 (`solar_open`).
- Ambiguous or strictness-dependent cases not counted below: `bamba`, `siglip2` L4 scope, SAM3 L4 scope, `pp_doclayout_v3`, `superpoint`, BART-style exact-L2 parity rows.
- Rationale-only defects: at least 15.

## §5: Files actually opened

This report is backed by direct parent-agent reads plus parallel read-only sub-audits. The highest-impact files opened by the parent include:

HF pinned source:
- `auto/modeling_auto.py:1-80`
- `donut/modeling_donut_swin.py:350-438`
- `higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py:1-648`
- `higgs_audio_v2_tokenizer/configuration_higgs_audio_v2_tokenizer.py:1-168`
- `rwkv/modeling_rwkv.py:1-758`
- `diffllama/modeling_diffllama.py:160-278`
- `mra/modeling_mra.py:1-100` plus class/function search reads
- `yoso/modeling_yoso.py:1-100` plus class/function search reads
- `dinat/modeling_dinat.py:1-120`
- `timm_backbone/modeling_timm_backbone.py:1-120`
- `timm_wrapper/modeling_timm_wrapper.py:1-120`
- `layoutlmv2/modeling_layoutlmv2.py:1-100`
- `xlstm/modeling_xlstm.py:1-120`
- `ibert/modeling_ibert.py:1-100`
- `fast_vlm/configuration_fast_vlm.py:1-112`, `fast_vlm/modeling_fast_vlm.py:1-180`
- `gemma3n/modular_gemma3n.py:300-439`, `gemma3n/modeling_gemma3n.py:130-349`
- `beit/modeling_beit.py:450-569`
- `bloom/modeling_bloom.py:140-307`
- `deberta/modeling_deberta.py:240-339`
- `deberta_v2/modeling_deberta_v2.py:80-345`
- `convbert/modeling_convbert.py:90-254`
- `encodec/modeling_encodec.py:70-169`
- `jais2/modular_jais2.py:1-107`
- `nemotron/modeling_nemotron.py:180-269`
- `vitdet/modeling_vitdet.py:130-249`
- `sam/modeling_sam.py:690-831`
- `qwen2_5_vl/modeling_qwen2_5_vl.py:1-220`
- `falcon_mamba/modeling_falcon_mamba.py:1-120`
- `xcodec/modeling_xcodec.py:1-220`
- `voxtral_realtime/modeling_voxtral_realtime.py:1-180`
- `vibevoice_acoustic_tokenizer/modeling_vibevoice_acoustic_tokenizer.py:1-220`
- `vibevoice_asr/modeling_vibevoice_asr.py:1-160`
- `siglip2/modeling_siglip2.py:1-220`
- `dinov3_vit/modeling_dinov3_vit.py:1-220`
- `sam3/modeling_sam3.py:1-120,2110-2289`

kb-nano source:
- `tasks/baseline/L1/rotary_emb.py:1-240`
- `tasks/baseline/L1/dense_attention.py:1-118`
- `tasks/baseline/L1/flash_attn_decode.py:1-80`
- `tasks/baseline/L1/conv1d.py:1-66`
- `tasks/baseline/L1/conv_transpose1d.py:1-55`
- `tasks/baseline/L1/squared_relu.py:1-15`
- `tasks/baseline/L1/squared_relu_and_mul.py:1-84`
- `tasks/baseline/L1/rms_norm.py:1-160`
- `tasks/baseline/L1/layer_norm.py:1-76`
- `tasks/baseline/L1/bitnet_linear.py:1-150`
- `tasks/baseline/L1/fp8_linear.py:1-120`
- `tasks/baseline/L1/rwkv7_recurrence.py:1-103`
- `tasks/baseline/L2/attention.py:1-260`
- `tasks/baseline/L2/encoder_attention.py:1-151`
- `tasks/baseline/L2/swinv2_window_attention.py:1-177`
- `tasks/baseline/L2/llama_mlp.py:1-38`
- `tasks/baseline/L2/encoder_mlp.py:1-34`
- `tasks/baseline/L2/clip_mlp.py:1-47`
- `tasks/baseline/L2/vision_attention.py`, `vision_patch_embed.py`, `vision_patch_merger.py`
- `tasks/baseline/L2/alphafold3_triangle_attention.py`, `alphafold3_of3_attention.py`, `alphafold3_triangle_multiplication.py`
- `tasks/baseline/L3/llama_decoder.py:1-42`
- `tasks/baseline/L4/qwen2_vl.py:1-220`
- `tasks/baseline/L4/qwen25_vl_encoder.py:1-200`
- `tasks/baseline/L4/mamba.py:1-180`
- `tasks/baseline/L4/rwkv7.py:1-120`
- `tasks/baseline/L4/dinov3.py:1-120`
- `tasks/baseline/L4/sam3.py:1-120,1130-1193`
- `tasks/baseline/L4/siglip2.py:1-140`

Sub-audits also opened the full source ranges summarized in their per-batch outputs for unsupported, L4, consistency flags, shard-trusted entries, rationale fixes, and partial/composable batches. I did not inline every subagent-read line range here because the set is hundreds of entries; the detailed report above lists the high-impact files and the per-finding evidence.

## §6: Confidence assessment

- High confidence wrong/fix-worthy: denominator identity; artifact/prose count drift; unsupported 12 stay unsupported; `solar_open` demotion over-applied; the 27 composable -> partial flips listed above; `edgetam` external-timm flip.
- Medium confidence / strictness-dependent: BART-style exact L2 parity, `siglip2` and SAM3 L4 scoping, `superpoint`, `pp_doclayout_v3`, `bamba`.
- Low confidence: none reported as a flip without caveat.

## §7: Pure ambiguities needing human judgment

- Should `torchaudio` in a modeling file count as unsupported external dependency? I chose yes for `higgs_audio_v2_tokenizer`; if no, classify it as `partial` instead.
- Should L4 status require full HF folder parity or allow a submodel L4 plus composable rest? This affects `siglip2` and SAM3 family L4 labels more than strict coverage.
- Should separate q/k/v BART-family attention be strict-partial because kb `whisper_attention.py` uses merged QKV, or composable because L1 Linear + attention can represent it? I did not count these as flips.
- `bamba`: config says partial rotary; runtime evidence did not confirm active partial slicing.

## §8: Things I did NOT fully verify

- I did not run any models or benchmarks; this is source-only.
- I did not personally hand-read every line of all 447 folders. I used parallel read-only sub-audits for full-batch coverage and personally spot-checked the highest-risk findings. The report is therefore strongest on the issue rows, not on asserting every unchanged row is perfect.
- Current GitHub `main` additions `hyperclovax` and `rf_detr` were only checked as source drift. They are not folded into the final 447-row headline.

## §9: New headline numbers if high-confidence fixes are applied

Using artifact counts as the starting point: 27 L4 / 238 composable / 170 partial / 12 unsupported.

Applying counted fixes:
- 27 composable -> partial flips.
- 1 composable -> unsupported flip (`edgetam`).
- 1 duplicate composable row (`donut`) replaced by missing `higgs_audio_v2_tokenizer` as unsupported.
- 1 partial -> composable flip (`solar_open`).

Suggested high-confidence headline:

- `kb_nano_l4`: 27
- `composable`: 210
- `partial`: 196
- `unsupported`: 14
- Strict: (27 + 210) / 447 = 237 / 447 = 53.02%
- Loose: (27 + 210 + 196) / 447 = 433 / 447 = 96.87%
- Unsupported: 14 / 447 = 3.13%

Alternative if `higgs_audio_v2_tokenizer` is counted as partial rather than unsupported:

- 27 / 210 / 197 / 13
- Strict: 237 / 447 = 53.02%
- Loose: 434 / 447 = 97.09%
- Unsupported: 13 / 447 = 2.91%
