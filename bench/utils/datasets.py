"""Dataset infrastructure for kb_nano benchmarks.

Re-exports vLLM's dataset types, and provides loaders for the standardized
text benchmark datasets: LongBench, ShareGPT, and DS-1000.
"""

from __future__ import annotations

import random

from vllm.benchmarks.datasets import (
    BenchmarkDataset,
    RandomDataset,
    RandomMultiModalDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    add_dataset_parser,
    add_random_dataset_base_args,
    add_random_multimodal_dataset_args,
    get_samples,
    process_image,
    process_video,
)


# ---------------------------------------------------------------------------
# Real-dataset loaders for text benchmarks
# ---------------------------------------------------------------------------

def load_longbench(tokenizer, num_requests: int, seed: int,
                   min_prompt_tokens: int = 12000) -> dict:
    """Load LongBench gov_report + multi_news, filter for long inputs.

    Returns {"prompts": list[str], "prompt_lens": list[int]}.
    """
    from datasets import load_dataset

    items = []
    for subset in ("gov_report", "multi_news"):
        ds = load_dataset("THUDM/LongBench", subset, split="test",
                          trust_remote_code=True)
        for row in ds:
            text = row.get("context", "") or row.get("input", "")
            if not text:
                continue
            items.append(text)

    rng = random.Random(seed)
    rng.shuffle(items)

    prompts = []
    prompt_lens = []
    for text in items:
        if len(prompts) >= num_requests:
            break
        ids = tokenizer.encode(text)
        if len(ids) >= min_prompt_tokens:
            prompts.append(text)
            prompt_lens.append(len(ids))

    if len(prompts) < num_requests:
        print(f"  WARNING: LongBench only yielded {len(prompts)} prompts "
              f"with >= {min_prompt_tokens} tokens (requested {num_requests})")

    return {"prompts": prompts, "prompt_lens": prompt_lens}


def load_sharegpt(tokenizer, num_requests: int, seed: int) -> dict:
    """Load ShareGPT single-turn conversations.

    Returns {"prompts": list[str], "prompt_lens": list[int],
             "output_lens": list[int]}.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split="train",
        trust_remote_code=True,
    )

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    prompts = []
    prompt_lens = []
    output_lens = []
    for idx in indices:
        if len(prompts) >= num_requests:
            break
        row = ds[idx]
        convs = row.get("conversations", [])
        if len(convs) < 2:
            continue
        prompt = convs[0].get("value", "")
        completion = convs[1].get("value", "")
        if not prompt or not completion:
            continue
        # Filter to reasonable lengths
        prompt_ids = tokenizer.encode(prompt)
        completion_ids = tokenizer.encode(completion)
        if len(prompt_ids) < 4 or len(prompt_ids) > 2048:
            continue
        if len(completion_ids) < 4:
            continue
        prompts.append(prompt)
        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(completion_ids))

    return {"prompts": prompts, "prompt_lens": prompt_lens,
            "output_lens": output_lens}


def load_ds1000(tokenizer, num_requests: int, seed: int) -> dict:
    """Load DS-1000 code generation prompts.

    Returns {"prompts": list[str], "prompt_lens": list[int]}.
    """
    from datasets import load_dataset

    ds = load_dataset("xlangai/DS-1000", split="test",
                      trust_remote_code=True)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    prompts = []
    prompt_lens = []
    for idx in indices:
        if len(prompts) >= num_requests:
            break
        row = ds[idx]
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        prompt_ids = tokenizer.encode(prompt)
        if len(prompt_ids) < 4:
            continue
        prompts.append(prompt)
        prompt_lens.append(len(prompt_ids))

    return {"prompts": prompts, "prompt_lens": prompt_lens}


def load_text_datasets(model: str, seed: int) -> dict:
    """Load all text datasets and return structured data.

    Returns dict with keys: "longbench", "sharegpt", "ds1000", each containing
    {"prompts": list[str], "prompt_lens": list[int],
     "output_lens": list[int] (sharegpt only)}.
    """
    from transformers import AutoTokenizer
    print("  Loading tokenizer and text datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    longbench = load_longbench(tokenizer, num_requests=500, seed=seed)
    print(f"    LongBench: {len(longbench['prompts'])} prompts loaded")

    sharegpt = load_sharegpt(tokenizer, num_requests=3000, seed=seed)
    print(f"    ShareGPT:  {len(sharegpt['prompts'])} prompts loaded")

    ds1000 = load_ds1000(tokenizer, num_requests=1000, seed=seed)
    print(f"    DS-1000:   {len(ds1000['prompts'])} prompts loaded")

    return {
        "longbench": longbench,
        "sharegpt": sharegpt,
        "ds1000": ds1000,
    }


__all__ = [
    "BenchmarkDataset",
    "RandomDataset",
    "RandomMultiModalDataset",
    "SampleRequest",
    "ShareGPTDataset",
    "SonnetDataset",
    "add_dataset_parser",
    "add_random_dataset_base_args",
    "add_random_multimodal_dataset_args",
    "get_samples",
    "process_image",
    "process_video",
    "load_longbench",
    "load_sharegpt",
    "load_ds1000",
    "load_text_datasets",
]
