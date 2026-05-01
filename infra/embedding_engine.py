"""vLLM-style pooling engine for encoder-only embedding models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

from .embedder_loader import load_bge_m3_model, load_colbertv2_model


@dataclass
class PoolingOutput:
    data: torch.Tensor


@dataclass
class PoolingRequestOutput:
    request_id: str
    outputs: PoolingOutput
    prompt_token_ids: list[int]
    num_cached_tokens: int
    finished: bool


class EmbeddingEngine:
    """Minimal vLLM-compatible pooling interface for kb-nano embedders."""

    def __init__(
        self,
        model_name: str,
        *,
        seed: int = 0,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda:0",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.seed = seed
        self.dtype = dtype
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        lower = model_name.lower()
        if "bge-m3" in lower or "bge_m3" in lower:
            self.model_key = "bge_m3"
            self.model, self.config = load_bge_m3_model(model_name, self.device, dtype)
        elif "colbert" in lower:
            self.model_key = "colbertv2"
            self.model, self.config = load_colbertv2_model(model_name, self.device, dtype)
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")

        torch.manual_seed(seed)

    def token_embed(
        self,
        prompts: str | Sequence[str],
        *,
        use_tqdm: bool = True,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        outputs = self.encode(
            prompts,
            pooling_task="token_embed",
            use_tqdm=use_tqdm,
            tokenization_kwargs=tokenization_kwargs,
        )
        return [item.outputs.data for item in outputs]

    def encode(
        self,
        prompts: str | Sequence[str],
        pooling_params: Any = None,
        *,
        use_tqdm: bool = True,
        pooling_task: str | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]:
        del pooling_params
        if pooling_task != "token_embed":
            raise ValueError('EmbeddingEngine.encode currently supports pooling_task="token_embed"')

        if isinstance(prompts, str):
            texts = [prompts]
        else:
            texts = list(prompts)

        tokenization_kwargs = dict(tokenization_kwargs or {})
        tokenization_kwargs.setdefault("truncation", True)

        results: list[PoolingRequestOutput] = []
        batch_starts = range(0, len(texts), self.batch_size)
        if use_tqdm:
            import sys

            from tqdm.auto import tqdm

            batch_starts = tqdm(
                batch_starts,
                total=(len(texts) + self.batch_size - 1) // self.batch_size,
                desc="kb-nano encode batches",
                unit="batch",
                file=sys.stdout,
            )
        with torch.no_grad():
            for start in batch_starts:
                batch_texts = texts[start:start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    return_tensors="pt",
                    **tokenization_kwargs,
                )
                prompt_token_ids = [
                    self.tokenizer(
                        text,
                        add_special_tokens=True,
                        padding=False,
                        verbose=False,
                        **tokenization_kwargs,
                    )["input_ids"]
                    for text in batch_texts
                ]
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                if self.model_key == "bge_m3":
                    hidden_states = self.model.forward_with_attention_mask(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        token_type_ids=encoded.get("token_type_ids"),
                    )
                    output = self.model.token_embed(hidden_states, encoded["attention_mask"])
                    masks = encoded["attention_mask"][:, 1:]
                else:
                    hidden_states = self.model.forward_with_attention_mask(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        token_type_ids=encoded.get("token_type_ids"),
                    )
                    output = self.model.token_embed(hidden_states, encoded["attention_mask"].bool())
                    masks = encoded["attention_mask"]

                for item, mask, token_ids in zip(output, masks, prompt_token_ids, strict=True):
                    valid = int(mask.sum().item())
                    results.append(
                        PoolingRequestOutput(
                            request_id=str(len(results)),
                            outputs=PoolingOutput(data=item[:valid].detach()),
                            prompt_token_ids=list(token_ids),
                            num_cached_tokens=0,
                            finished=True,
                        )
                    )

        return results
