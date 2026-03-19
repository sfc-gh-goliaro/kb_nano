"""Eval planner: static job generation from (model, tp) pairs.

Given an EvalConfig, produces an EvalPlan — an ordered list of E2E job
pairs (baseline + candidate). Each job runs all standardized workloads
in a single engine load using real datasets (LongBench, ShareGPT, DS-1000).

Jobs are ordered by TP degree (ascending) for optimal GPU scheduling:
TP=1 jobs can run 8 in parallel on an 8-GPU box, TP=4 jobs run 2 in
parallel, TP=8 jobs run 1 at a time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from kb_nano.bench.utils.workloads import (
    LATENCY_WORKLOADS,
    THROUGHPUT_WORKLOADS,
)

from .config import EvalConfig, MODEL_KEY_TO_DEFAULT_HF


@dataclass
class EvalJob:
    """A single E2E benchmark job (one model, one TP degree, all workloads)."""
    model: str
    tp: int
    category: str
    seed: int = 42
    temperature: float = 0.0
    enforce_eager: bool = False

    throughput_data: list[dict] = field(default_factory=list)
    latency_data: list[dict] = field(default_factory=list)

    @property
    def short_name(self) -> str:
        return self.model.split("/")[-1]


@dataclass
class EvalPlan:
    """Ordered list of eval jobs, sorted by TP degree for GPU scheduling."""
    jobs: list[EvalJob] = field(default_factory=list)
    max_seq_len: int = 0

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    def models_by_category(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for job in self.jobs:
            result.setdefault(job.category, []).append(
                f"{job.short_name} (TP={job.tp})"
            )
        return result


class EvalPlanner:
    """Generates an EvalPlan from EvalConfig."""

    def __init__(self, config: EvalConfig):
        self.config = config

    def _resolve_models(self) -> list[str]:
        """Determine which HF models to evaluate."""
        if self.config.models:
            return self.config.models

        from kb_nano.infra.kernel_swapper import discover_candidates
        candidates = discover_candidates()
        if not candidates:
            return []

        model_keys: set[str] = set()
        for target, _ in candidates:
            model_keys.update(target.models)

        models = []
        for key in sorted(model_keys):
            if key in MODEL_KEY_TO_DEFAULT_HF:
                models.append(MODEL_KEY_TO_DEFAULT_HF[key])
        return models

    def _generate_workload_data(
        self,
        model: str,
        seed: int,
    ) -> tuple[list[dict], list[dict], int]:
        """Load real datasets and generate workload data.

        Returns (throughput_data, latency_data, max_seq_len).
        """
        from kb_nano.bench.utils.datasets import (
            load_longbench, load_sharegpt, load_ds1000,
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        datasets = {
            "longbench": load_longbench(tokenizer, num_requests=500, seed=seed),
            "sharegpt": load_sharegpt(tokenizer, num_requests=3000, seed=seed),
            "ds1000": load_ds1000(tokenizer, num_requests=1000, seed=seed),
        }

        max_seq_len = 0

        throughput_data = []
        for w in THROUGHPUT_WORKLOADS:
            src = datasets[w.dataset]
            n = min(w.num_requests, len(src["prompts"]))
            prompts = src["prompts"][:n]
            prompt_lens = src["prompt_lens"][:n]

            if w.output_len is not None:
                output_lens = [w.output_len] * n
            else:
                output_lens = src["output_lens"][:n]

            max_prompt_len = max(prompt_lens) if prompt_lens else 0
            max_output_len = max(output_lens) if output_lens else 0
            seq_len = max_prompt_len + max_output_len
            if seq_len > max_seq_len:
                max_seq_len = seq_len

            throughput_data.append({
                "name": w.name,
                "prompts": prompts,
                "output_lens": output_lens,
                "num_requests": n,
                "avg_prompt_len": sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0,
                "max_prompt_len": max_prompt_len,
            })

        latency_data = []
        for w in LATENCY_WORKLOADS:
            src = datasets[w.dataset]
            # Pick from end to avoid overlap with throughput
            prompts = src["prompts"][-w.batch_size:]
            prompt_lens = src["prompt_lens"][-w.batch_size:]

            max_prompt_len = max(prompt_lens) if prompt_lens else 0
            seq_len = max_prompt_len + w.output_len
            if seq_len > max_seq_len:
                max_seq_len = seq_len

            latency_data.append({
                "name": w.name,
                "output_len": w.output_len,
                "batch_size": w.batch_size,
                "prompts": prompts,
                "num_warmup": w.num_warmup,
                "num_iters": w.num_iters,
            })

        return throughput_data, latency_data, max_seq_len

    def plan(self) -> EvalPlan:
        """Generate the eval plan: ordered list of (model, tp) job pairs."""
        models = self._resolve_models()
        if not models:
            return EvalPlan()

        if self.config.categories:
            models = [
                m for m in models
                if self.config.get_model_category(m) in self.config.categories
            ]

        # Use the first model to load datasets (tokenizer needed for length
        # estimation). All models use the same real-text prompts.
        first_model = models[0] if models else "meta-llama/Llama-3.1-8B-Instruct"
        throughput_data, latency_data, max_seq_len = self._generate_workload_data(
            first_model, self.config.seed,
        )

        jobs: list[EvalJob] = []
        for model in models:
            category = self.config.get_model_category(model)
            for tp in sorted(self.config.tp_degrees):
                jobs.append(EvalJob(
                    model=model,
                    tp=tp,
                    category=category,
                    seed=self.config.seed,
                    temperature=self.config.temperature,
                    enforce_eager=self.config.enforce_eager,
                    throughput_data=throughput_data,
                    latency_data=latency_data,
                ))

        jobs.sort(key=lambda j: j.tp)

        return EvalPlan(jobs=jobs, max_seq_len=max_seq_len)
