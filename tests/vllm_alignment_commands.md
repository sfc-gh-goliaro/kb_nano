# vLLM Alignment Test Commands

## Llama 3.1 8B Instruct (`ref-only`)

```bash
python tests/test_vllm_alignment.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --ref-only \
    --max-tokens 64 \
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

## Run Log

- `2026-03-11`: `meta-llama/Llama-4-Scout-17B-16E-Instruct` command succeeded (`PASS`).

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
