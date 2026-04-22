# vLLM Alignment Test Commands

## GPT-OSS-20B (`ref-only`)

```bash
python tests/test_vllm_alignment.py \
    --model openai/gpt-oss-20b \
    --ref-only \
    --tp 1 \
    --max-tokens 32 \
    --seed 42
```

## Mamba-Codestral-7B v0.1 (`ref-only`, force HF load format)

```bash
python tests/test_vllm_alignment.py \
    --model mistralai/Mamba-Codestral-7B-v0.1 \
    --ref-only \
    --tp 1 \
    --max-tokens 32 \
    --seed 42 \
    --trust-remote-code \
    --load-format hf
```

## Llama 3.1 8B Instruct (`ref-only`)

```bash
python tests/test_vllm_alignment.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --ref-only \
    --max-tokens 64 \
    --seed 42
```

## Kimi Linear 48B A3B Instruct (`ref-only`)

```bash
python tests/test_vllm_alignment.py \
    --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
    --ref-only \
    --trust-remote-code \
    --tp 4 \
    --max-tokens 32 \
    --seed 42
```

## Kimi Linear 48B A3B Instruct (ours vs vLLM)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_vllm_alignment.py \
    --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
    --trust-remote-code \
    --tp 4 \
    --max-tokens 32 \
    --seed 42
```

- `2026-03-12`: 2/4 exact matches. Divergences at tokens 0 and 20 due to kernel numerical differences (SDPA vs Flash Attention MLA, different Triton MoE kernels).

## GPT-OSS-20B (ours vs vLLM)

```bash
python tests/test_vllm_alignment.py \
    --model openai/gpt-oss-20b \
    --tp 1 \
    --max-tokens 32 \
    --seed 42
```

- `2026-03-12`: 3/4 exact matches at 32 tokens, 4/4 at 15 tokens. Minor divergences due to SDPA vs FlashAttention numerical differences.

## Qwen3-Next 80B A3B Instruct (`ref-only`)

```bash
python tests/test_vllm_alignment.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --ref-only \
    --trust-remote-code \
    --tp 4 \
    --max-tokens 32 \
    --seed 42
```

## Llama-4 Scout 17B 16E Instruct (`ref-only`)

```bash
python tests/test_vllm_alignment.py \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --ref-only \
    --tp 4 \
    --max-tokens 32 \
    --seed 42
```

## Native BitNet Alignment (`ref-only`)

```bash
python tests/test_native_bitnet_alignment.py \
    --ref-only \
    --max-tokens 64 \
    --seed 42
```

## Native BitNet Alignment (standalone vs GPU FastGen)

```bash
python tests/test_native_bitnet_alignment.py \
    --max-tokens 8 \
    --seed 42
```

- `2026-03-12`: standalone vs GPU FastGen alignment succeeded (3/4 `PASS`).

## Native BitNet Standalone Only

```bash
python tests/test_native_bitnet_alignment.py \
    --standalone-only \
    --max-tokens 32 \
    --seed 42
```

## Run Log

- `2026-03-11`: `Qwen/Qwen3-Next-80B-A3B-Instruct` first command succeeded (`PASS`).
- `2026-03-11`: `meta-llama/Llama-4-Scout-17B-16E-Instruct` command succeeded (`PASS`).
- `2026-03-11`: `test_native_bitnet_alignment.py --ref-only --max-tokens 64 --seed 42` succeeded (`PASS`).
- `2026-03-12`: `mistralai/Mamba-Codestral-7B-v0.1` with `--load-format hf` succeeded (`PASS`).

## vLLM EAGLE Ref-Only Alignment (Acceptance Report)

```bash
python tests/test_vllm_eagle_alignment.py \
    --target-model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model yuhuili/EAGLE-LLaMA3.1-Instruct-8B \
    --method eagle \
    --tp 1 \
    --draft-tp 1 \
    --num-speculative-tokens 4 \
    --max-tokens 64 \
    --seed 42
```

- `2026-03-11`: `test_vllm_eagle_alignment.py` with Llama3.1 + EAGLE (num-spec-tokens=4) succeeded (`PASS`).

## vLLM EAGLE Explicit Forward (Manual Loop, `ref-only`)

```bash
python tests/test_vllm_eagle_alignment.py \
    --target-model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model yuhuili/EAGLE-LLaMA3.1-Instruct-8B \
    --method eagle \
    --tp 1 \
    --draft-tp 1 \
    --num-speculative-tokens 4 \
    --max-tokens 64 \
    --seed 42
```

- `2026-03-12`: explicit-forward `test_vllm_eagle_alignment.py` (Llama3.1 + EAGLE, target on `cuda:1`) succeeded (`PASS`).

## Native FLA GLA-2.7B (`ref-only`)

```bash
python tests/test_native_fla_alignment.py \
    --model fla-hub/gla-2.7B-100B \
    --ref-only \
    --max-tokens 64 \
    --seed 42
```

- `2026-03-12`: `test_native_fla_alignment.py` with `fla-hub/gla-2.7B-100B` succeeded (`PASS`).

## Native FLA GLA-2.7B Alignment (ours vs FLA)

```bash
python tests/test_native_fla_alignment.py \
    --model fla-hub/gla-2.7B-100B \
    --max-tokens 32 \
    --seed 42
```

## Native FLA RWKV7-2.9B (`ref-only`)

```bash
python tests/test_native_fla_alignment.py \
    --model fla-hub/rwkv7-2.9B-world \
    --ref-only \
    --max-tokens 64 \
    --seed 42
```

- `2026-03-12`: `test_native_fla_alignment.py` with `fla-hub/rwkv7-2.9B-world` succeeded (`PASS`).

## Native FLA RWKV7-2.9B Alignment (ours vs FLA)

```bash
python tests/test_native_fla_alignment.py \
    --model fla-hub/rwkv7-2.9B-world \
    --max-tokens 32 \
    --seed 42
```

## Native FLA RetNet-2.7B (`ref-only`)

```bash
python tests/test_native_fla_alignment.py \
    --model fla-hub/retnet-2.7B-100B \
    --ref-only \
    --max-tokens 64 \
    --seed 42
```

- `2026-03-12`: `test_native_fla_alignment.py` with `fla-hub/retnet-2.7B-100B` succeeded (`PASS`).

## Native FLA RetNet-2.7B Alignment (ours vs FLA)

```bash
python tests/test_native_fla_alignment.py \
    --model fla-hub/retnet-2.7B-100B \
    --max-tokens 32 \
    --seed 42
```
