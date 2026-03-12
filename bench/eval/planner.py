"""Eval planner: static job generation from (model, tp) pairs.

Given an EvalConfig, produces an EvalPlan — an ordered list of E2E job
pairs (baseline + candidate). Each job runs all 5 standardized workloads
in a single engine load.

Jobs are ordered by TP degree (ascending) for optimal GPU scheduling:
TP=1 jobs can run 8 in parallel on an 8-GPU box, TP=4 jobs run 2 in
parallel, TP=8 jobs run 1 at a time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from random import randint

from kb_nano.bench.utils.workloads import (
    LATENCY_WORKLOADS,
    THROUGHPUT_WORKLOADS,
    get_max_seq_len,
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
        seed: int,
        num_prompts: int,
    ) -> tuple[list[dict], list[dict]]:
        """Pre-generate all scenario data (matching bench_vllm.py methodology)."""
        throughput_data = []
        for i, w in enumerate(THROUGHPUT_WORKLOADS):
            rng_seed = seed + i
            random.seed(rng_seed)
            prompt_token_ids = [
                [randint(0, 10000) for _ in range(w.input_len)]
                for _ in range(num_prompts)
            ]
            output_lens = [w.output_len] * num_prompts
            throughput_data.append({
                "name": w.name,
                "input_len": w.input_len,
                "output_len": w.output_len,
                "prompt_token_ids": prompt_token_ids,
                "output_lens": output_lens,
            })

        latency_data = []
        for j, w in enumerate(LATENCY_WORKLOADS):
            rng_seed = seed + 100 + j
            random.seed(rng_seed)
            prompt_token_ids = [
                [randint(0, 10000) for _ in range(w.input_len)]
                for _ in range(w.batch_size)
            ]
            latency_data.append({
                "name": w.name,
                "input_len": w.input_len,
                "output_len": w.output_len,
                "batch_size": w.batch_size,
                "prompt_token_ids": prompt_token_ids,
                "num_warmup": w.num_warmup,
                "num_iters": w.num_iters,
            })

        return throughput_data, latency_data

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

        max_seq_len = get_max_seq_len()
        throughput_data, latency_data = self._generate_workload_data(
            self.config.seed, self.config.num_prompts,
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
