# Cross-agent consistency audit

> Historical document (Phase 1 of the audit). The "final = v4" annotation
> below was the state after first-pass + cross-verifier rounds; the audit
> later went through v7, v10, v11, and v12 (canonical). For final v12
> numbers see `REAUDIT_NOTES.md` and `CAVEATS_AND_METHODOLOGY.md`.

(based on 425 first-pass folders + 239 cross-verifier touches; final = v4)

## Pattern groups: agent verdicts side-by-side

Format: folder | first-pass | cross-verifier | final (v4)


### Pattern: `T5_cross_attention` (92 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable', 'kb_nano_l4'}, final: {'unsupported', 'partial', 'composable', 'kb_nano_l4'}

  ✓ `aria` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `bart` — fp:composable  | xv:-                         | final:composable
  ✓ `bert` — fp:composable  | xv:-                         | final:composable
  ✓ `bert_generation` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `bigbird_pegasus` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `biogpt` — fp:composable  | xv:-                         | final:composable
  ✓ `blenderbot` — fp:composable  | xv:-                         | final:composable
  ✓ `blenderbot_small` — fp:composable  | xv:-                         | final:composable
  ✓ `blip/blip_text` — fp:composable  | xv:-                         | final:composable
  ✓ `blip_2` — fp:composable  | xv:-                         | final:composable
  ✓ `blt` — fp:composable  | xv:-                         | final:composable
  ✓ `bridgetower` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `camembert` — fp:composable  | xv:-                         | final:composable
  ✓ `canine` — fp:composable  | xv:-                         | final:composable
  ✓ `cohere_asr` — fp:composable  | xv:-                         | final:composable
  ✓ `d_fine` — fp:composable  | xv:-                         | final:composable
  ✓ `dab_detr` — fp:composable  | xv:-                         | final:composable
  ✓ `data2vec_text` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `detr` — fp:composable  | xv:-                         | final:composable
  ✓ `dia` — fp:composable  | xv:-                         | final:composable
  ✓ `edgetam` — fp:composable  | xv:-                         | final:composable
  ✓ `edgetam_video` — fp:composable  | xv:-                         | final:composable
  ✓ `efficientloftr` — fp:composable  | xv:-                         | final:composable
  ✓ `electra` — fp:composable  | xv:-                         | final:composable
  ✓ `ernie` — fp:composable  | xv:-                         | final:composable
  ⚠ `evolla` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `flaubert` — fp:unsupported | xv:confirm_partial           | final:partial
  ⚠ `fsmt` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `gemma3n` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ✓ `gpt2` — fp:composable  | xv:-                         | final:composable
  ⚠ `granite_speech` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `grounding_dino` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `groupvit` — fp:composable  | xv:downgrade_to_partial      | final:partial
  ✓ `idefics` — fp:composable  | xv:-                         | final:composable
  ✓ `idefics2` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `imagegpt` — fp:composable  | xv:-                         | final:composable
  ✓ `instructblip` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `instructblipvideo` — fp:composable  | xv:-                         | final:composable
  ✓ `kosmos2_5` — fp:composable  | xv:-                         | final:composable
  ⚠ `lightglue` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `lilt` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `lxmert` — fp:composable  | xv:-                         | final:composable
  ✓ `m2m_100` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `marian` — fp:composable  | xv:-                         | final:composable
  ⚠ `maskformer` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `mbart` — fp:composable  | xv:-                         | final:composable
  ✓ `mllama` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `moonshine` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `mt5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `musicgen` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `musicgen_melody` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `nllb_moe` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `omdet_turbo` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `oneformer` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `patchtst` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `pegasus` — fp:composable  | xv:-                         | final:composable
  ✓ `pegasus_x` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `perceiver` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `pix2struct` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `plbart` — fp:composable  | xv:-                         | final:composable
  ✓ `pop2piano` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `pp_doclayout_v2` — fp:composable  | xv:-                         | final:composable
  ✓ `pp_formulanet` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `prophetnet` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `roberta` — fp:composable  | xv:-                         | final:composable
  ✓ `roberta_prelayernorm` — fp:composable  | xv:-                         | final:composable
  ✓ `roc_bert` — fp:composable  | xv:-                         | final:composable
  ✓ `sam` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `sam2` — fp:composable  | xv:-                         | final:composable
  ✓ `sam2_video` — fp:composable  | xv:-                         | final:composable
  ✓ `sam3` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `sam3_tracker` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `sam3_tracker_video` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ⚠ `sam_hq` — fp:composable  | xv:downgrade_to_partial      | final:partial
  ✓ `speech_to_text` — fp:composable  | xv:-                         | final:composable
  ✓ `superglue` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `switch_transformers` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `t5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `t5gemma` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `t5gemma2` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `table_transformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `time_series_transformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `trocr` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `udop` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `umt5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `visual_bert` — fp:composable  | xv:-                         | final:composable
  ✓ `vjepa2` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `whisper` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `x_clip` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `xglm` — fp:composable  | xv:-                         | final:composable
  ✓ `xlm_roberta` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `xmod` — fp:composable  | xv:-                         | final:composable

### Pattern: `LayerNorm_decoder` (49 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable', 'kb_nano_l4'}, final: {'unsupported', 'partial', 'composable', 'kb_nano_l4'}

  ✓ `albert` — fp:composable  | xv:-                         | final:composable
  ✓ `autoformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `bert` — fp:composable  | xv:-                         | final:composable
  ✓ `bert_generation` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `bridgetower` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `camembert` — fp:composable  | xv:-                         | final:composable
  ✓ `canine` — fp:composable  | xv:-                         | final:composable
  ✓ `chameleon` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `cohere` — fp:composable  | xv:-                         | final:composable
  ✓ `cohere_asr` — fp:composable  | xv:-                         | final:composable
  ✓ `d_fine` — fp:composable  | xv:-                         | final:composable
  ✓ `dab_detr` — fp:composable  | xv:-                         | final:composable
  ✓ `deimv2` — fp:composable  | xv:-                         | final:composable
  ✓ `deit` — fp:composable  | xv:-                         | final:composable
  ✓ `dia` — fp:composable  | xv:-                         | final:composable
  ✓ `dinat` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ⚠ `evolla` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `fuyu` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `hiera` — fp:composable  | xv:-                         | final:composable
  ✓ `idefics` — fp:composable  | xv:-                         | final:composable
  ✓ `lfm2` — fp:composable  | xv:-                         | final:composable
  ✓ `moonshine_streaming` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `olmo` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `opt` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `persimmon` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `phi` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `phimoe` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `pix2struct` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `pp_doclayout_v2` — fp:composable  | xv:-                         | final:composable
  ✓ `pp_formulanet` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `qwen2_5_omni` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `qwen3_omni_moe` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `rembert` — fp:composable  | xv:-                         | final:composable
  ✓ `roberta` — fp:composable  | xv:-                         | final:composable
  ✓ `roberta_prelayernorm` — fp:composable  | xv:-                         | final:composable
  ✓ `roc_bert` — fp:composable  | xv:-                         | final:composable
  ✓ `roformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `sam` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `sam2` — fp:composable  | xv:-                         | final:composable
  ✓ `sam2_video` — fp:composable  | xv:-                         | final:composable
  ✓ `sam3_tracker` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `sam3_tracker_video` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ⚠ `sam_hq` — fp:composable  | xv:downgrade_to_partial      | final:partial
  ✓ `seggpt` — fp:composable  | xv:-                         | final:composable
  ✓ `slanext` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `stablelm` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `swin` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `time_series_transformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `videomae` — fp:composable  | xv:-                         | final:composable

### Pattern: `weight_norm` (42 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable', 'kb_nano_l4'}, final: {'partial', 'composable', 'kb_nano_l4'}

  ✓ `afmoe` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `bit` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `bitnet` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `dbrx` — fp:composable  | xv:-                         | final:composable
  ⚠ `deepseek_v4` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `deformable_detr` — fp:composable  | xv:-                         | final:composable
  ✓ `doge` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `encodec` — fp:composable  | xv:-                         | final:composable
  ⚠ `fastspeech2_conformer` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `gemma` — fp:composable  | xv:-                         | final:composable
  ✓ `glm_moe_dsa` — fp:composable  | xv:-                         | final:composable
  ✓ `granitemoe` — fp:composable  | xv:-                         | final:composable
  ✓ `hubert` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `imagegpt` — fp:composable  | xv:-                         | final:composable
  ✓ `jetmoe` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `kyutai_speech_to_text` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `lfm2_moe` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `llama4` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `mimi` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `mobilebert` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `moshi` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `nanochat` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `nemotron` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `olmo` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `olmo2` — fp:composable  | xv:-                         | final:composable
  ✓ `olmo_hybrid` — fp:composable  | xv:-                         | final:composable
  ✓ `qwen3_5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `qwen3_next` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `recurrent_gemma` — fp:composable  | xv:-                         | final:composable
  ✓ `seamless_m4t` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `sew` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `t5` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `timesfm` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `unispeech` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `unispeech_sat` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `univnet` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `vaultgemma` — fp:composable  | xv:-                         | final:composable
  ✓ `videomt` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `vits` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `wav2vec2` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `wavlm` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `zamba` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `BART_cross` (40 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable'}, final: {'partial', 'composable'}

  ✓ `bart` — fp:composable  | xv:-                         | final:composable
  ✓ `biogpt` — fp:composable  | xv:-                         | final:composable
  ✓ `blenderbot` — fp:composable  | xv:-                         | final:composable
  ✓ `blenderbot_small` — fp:composable  | xv:-                         | final:composable
  ⚠ `flaubert` — fp:unsupported | xv:confirm_partial           | final:partial
  ⚠ `florence2` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `fsmt` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `hubert` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `idefics` — fp:composable  | xv:-                         | final:composable
  ⚠ `informer` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `kosmos2` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `kosmos2_5` — fp:composable  | xv:-                         | final:composable
  ⚠ `led` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `m2m_100` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `marian` — fp:composable  | xv:-                         | final:composable
  ✓ `mbart` — fp:composable  | xv:-                         | final:composable
  ✓ `mt5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `musicgen` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `musicgen_melody` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `mvp` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `nllb_moe` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `pegasus` — fp:composable  | xv:-                         | final:composable
  ✓ `plbart` — fp:composable  | xv:-                         | final:composable
  ✓ `pp_formulanet` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `prophetnet` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `seamless_m4t` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `seamless_m4t_v2` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `sew` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `speech_encoder_decoder` — fp:composable  | xv:-                         | final:composable
  ✓ `speecht5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `t5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `table_transformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `time_series_transformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `trocr` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `unispeech` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `unispeech_sat` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `vision_encoder_decoder` — fp:composable  | xv:-                         | final:composable
  ✓ `wav2vec2` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `xglm` — fp:composable  | xv:-                         | final:composable
  ✓ `xlm` — fp:composable  | xv:-                         | final:composable

### Pattern: `Conformer_rel_shift` (32 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable'}, final: {'unsupported', 'partial', 'composable'}

  ✓ `beit` — fp:composable  | xv:-                         | final:composable
  ✓ `clap` — fp:composable  | xv:-                         | final:composable
  ✓ `cpmant` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `data2vec_vision` — fp:composable  | xv:-                         | final:composable
  ✓ `donut` — fp:composable  | xv:-                         | final:composable
  ⚠ `fastspeech2_conformer` — fp:unsupported | xv:confirm_partial           | final:partial
  ⚠ `funnel` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `gemma3n` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ⚠ `glmasr` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `granite_speech` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `lasr` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `layoutlmv2` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ⚠ `longt5` — fp:unsupported | xv:confirm_partial           | final:partial
  ⚠ `maskformer_swin` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `mpnet` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `mt5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `parakeet` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `phi4_multimodal` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `pix2struct` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `pp_doclayout_v2` — fp:composable  | xv:-                         | final:composable
  ✓ `pp_formulanet` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `prophetnet` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `seamless_m4t` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `seamless_m4t_v2` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `speecht5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `swin` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `udop` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `vitdet` — fp:composable  | xv:-                         | final:composable
  ⚠ `wav2vec2_bert` — fp:unsupported | xv:confirm_partial           | final:partial
  ⚠ `wav2vec2_conformer` — fp:unsupported | xv:confirm_partial           | final:partial
  ⚠ `wavlm` — fp:unsupported | xv:confirm_partial           | final:partial
  ⚠ `xlnet` — fp:unsupported | xv:confirm_partial           | final:partial

### Pattern: `sliding_chunked` (22 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable', 'kb_nano_l4'}, final: {'partial', 'composable', 'kb_nano_l4'}

  ✓ `afmoe` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `big_bird` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `bigbird_pegasus` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `cohere2` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `cwm` — fp:composable  | xv:confirm_composable        | final:composable
  ⚠ `deepseek_v4` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `gpt_neo` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `gpt_oss` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ⚠ `led` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `mistral` — fp:composable  | xv:-                         | final:composable
  ✓ `mixtral` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `modernbert` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `modernbert_decoder` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `moonshine_streaming` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `olmo3` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `openai_privacy_filter` — fp:composable  | xv:-                         | final:composable
  ✓ `qwen2` — fp:composable  | xv:-                         | final:composable
  ✓ `qwen3_moe` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `reformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `smollm3` — fp:composable  | xv:-                         | final:composable
  ✓ `starcoder2` — fp:composable  | xv:-                         | final:composable
  ✓ `t5gemma` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `AutoBackbone` (13 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable'}, final: {'partial', 'composable'}

  ⚠ `chmv2` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `conditional_detr` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `dab_detr` — fp:composable  | xv:-                         | final:composable
  ✓ `deformable_detr` — fp:composable  | xv:-                         | final:composable
  ✓ `depth_anything` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `detr` — fp:composable  | xv:-                         | final:composable
  ✓ `grounding_dino` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `modernvbert` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `omdet_turbo` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `oneformer` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `prompt_depth_anything` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `table_transformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `tvp` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `BatchNorm1d` (11 folders)

  ⚠ `fastspeech2_conformer` — fp:unsupported | xv:confirm_partial           | final:partial
  ⚠ `glmasr` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ⚠ `granite_speech` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `hubert` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `informer` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `lasr` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `levit` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `patchtsmixer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `patchtst` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `speecht5` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `superglue` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `nn_MultiheadAttention` (9 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable'}, final: {'partial', 'composable'}

  ✓ `aria` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `bridgetower` — fp:composable  | xv:confirm_composable        | final:composable
  ✓ `ctrl` — fp:composable  | xv:-                         | final:composable
  ⚠ `flaubert` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `idefics2` — fp:composable  | xv:confirm_composable        | final:composable
  ⚠ `oneformer` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `pp_formulanet` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `xlm` — fp:composable  | xv:-                         | final:composable
  ✓ `zoedepth` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `partial_rotary` (7 folders)

  ✓ `codegen` — fp:composable  | xv:-                         | final:composable
  ✓ `gpt_neox` — fp:composable  | xv:-                         | final:composable
  ✓ `gptj` — fp:composable  | xv:-                         | final:composable
  ✓ `persimmon` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `phi` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `phi3` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `stablelm` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `Snake1d_or_xIELU` (5 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'kb_nano_l4'}, final: {'partial', 'kb_nano_l4'}

  ⚠ `apertus` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `dac` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `pe_audio` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `qwen2_5_omni` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `qwen3_omni_moe` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `mamba_variant` (4 folders)

  ✓ `bamba` — fp:composable  | xv:-                         | final:composable
  ⚠ `falcon_mamba` — fp:kb_nano_l4  | xv:confirm_composable        | final:composable
  ✓ `jamba` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4
  ✓ `mamba` — fp:kb_nano_l4  | xv:confirm_l4                | final:kb_nano_l4

### Pattern: `ALiBi` (4 folders)

**MIXED VERDICTS** — first-pass: {'unsupported', 'partial', 'composable'}, final: {'unsupported', 'partial'}

  ⚠ `bloom` — fp:composable  | xv:confirm_partial           | final:partial
  ⚠ `falcon` — fp:unsupported | xv:confirm_partial           | final:partial
  ✓ `layoutlmv2` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ✓ `mpt` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `kernels_community` (4 folders)

  ⚠ `deepseek_v4` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `mra` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ⚠ `omdet_turbo` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `yoso` — fp:unsupported | xv:confirm_unsupported       | final:unsupported

### Pattern: `timm_dep` (3 folders)

  ✓ `fast_vlm` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ✓ `timm_backbone` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ✓ `timm_wrapper` — fp:unsupported | xv:confirm_unsupported       | final:unsupported

### Pattern: `clip_qkv` (3 folders)

  ✓ `olmo` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `olmo2` — fp:composable  | xv:-                         | final:composable
  ✓ `olmoe` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `nn_GRUCell` (3 folders)

  ✓ `slanet` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `slanext` — fp:partial     | xv:confirm_partial           | final:partial
  ⚠ `wavlm` — fp:unsupported | xv:confirm_partial           | final:partial

### Pattern: `torch_fft` (2 folders)

  ✓ `autoformer` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `fnet` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `Snake1d_only` (2 folders)

  ✓ `dac` — fp:partial     | xv:confirm_partial           | final:partial
  ✓ `pe_audio` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `detectron2` (2 folders)

  ✓ `layoutlmv2` — fp:unsupported | xv:confirm_unsupported       | final:unsupported
  ✓ `layoutlmv3` — fp:partial     | xv:confirm_partial           | final:partial

### Pattern: `autograd_Function` (2 folders)

  ⚠ `phimoe` — fp:unsupported | xv:downgrade_to_partial      | final:partial
  ✓ `reformer` — fp:partial     | xv:confirm_partial           | final:partial


## Summary

- patterns analyzed: 21
- patterns with mixed verdicts: 10
- total folder-touches across agents: 425 (first-pass) + 355 pattern-hits + 239 cross-verified