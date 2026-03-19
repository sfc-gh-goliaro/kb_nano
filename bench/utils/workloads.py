"""Standardized workload definitions for eval (Tier 3).

These workloads are constants that ensure reproducible, comparable results
across runs and users. They are not configurable by design.

LLM workloads (real datasets):
  Throughput: 3 scenarios using LongBench (500 reqs, prefill-heavy),
    ShareGPT (3000 reqs, decode-heavy at high concurrency), and
    DS-1000 (1000 reqs, decode-sustained code generation).
  Latency: 2 scenarios (single-short from ShareGPT, single-long-context
    from LongBench), both batch_size=1.

VLM workloads (multi-modal):
  Throughput: 3 scenarios (text-only mixed from LLM datasets, image, video).
  Latency: 2 scenarios (single-image, single-video), batch_size=1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThroughputWorkload:
    name: str
    dataset: str  # "longbench", "sharegpt", "ds1000"
    output_len: int | None  # None = dynamic (from dataset)
    num_requests: int = 1000


@dataclass(frozen=True)
class LatencyWorkload:
    name: str
    dataset: str  # "sharegpt", "longbench"
    batch_size: int
    output_len: int
    num_warmup: int = 3
    num_iters: int = 5


THROUGHPUT_WORKLOADS: list[ThroughputWorkload] = [
    ThroughputWorkload(
        name="longbench-summ", dataset="longbench",
        output_len=512, num_requests=500),
    ThroughputWorkload(
        name="sharegpt-short", dataset="sharegpt",
        output_len=None, num_requests=3000),
    ThroughputWorkload(
        name="ds1000-code", dataset="ds1000",
        output_len=8192, num_requests=1000),
]

LATENCY_WORKLOADS: list[LatencyWorkload] = [
    LatencyWorkload(
        name="single-short", dataset="sharegpt",
        batch_size=1, output_len=128),
    LatencyWorkload(
        name="single-long-context", dataset="longbench",
        batch_size=1, output_len=128),
]

ALL_WORKLOADS = {
    "throughput": THROUGHPUT_WORKLOADS,
    "latency": LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# VLM workloads (multi-modal)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VLMThroughputWorkload:
    name: str
    modality: str  # "text", "image", "video"
    output_len: int | None  # None = dynamic (from dataset)
    dataset_name: str | None = None  # HF dataset (image/video only)
    num_requests: int = 1000


VLM_THROUGHPUT_WORKLOADS: list[VLMThroughputWorkload] = [
    VLMThroughputWorkload(
        "text-only", "text", output_len=None,
        dataset_name="mixed"),
    VLMThroughputWorkload(
        "image", "image", output_len=512,
        dataset_name="lmarena-ai/VisionArena-Chat"),
    VLMThroughputWorkload(
        "video", "video", output_len=512,
        dataset_name="yale-nlp/MMVU"),
]


@dataclass(frozen=True)
class VLMLatencyWorkload:
    name: str
    modality: str  # "image", "video"
    output_len: int
    batch_size: int = 1
    dataset_name: str | None = None
    num_warmup: int = 3
    num_iters: int = 5


VLM_LATENCY_WORKLOADS: list[VLMLatencyWorkload] = [
    VLMLatencyWorkload(
        "single-image", "image", output_len=128,
        dataset_name="lmarena-ai/VisionArena-Chat"),
    VLMLatencyWorkload(
        "single-video", "video", output_len=128,
        dataset_name="yale-nlp/MMVU"),
]

ALL_VLM_WORKLOADS = {
    "throughput": VLM_THROUGHPUT_WORKLOADS,
    "latency": VLM_LATENCY_WORKLOADS,
}
