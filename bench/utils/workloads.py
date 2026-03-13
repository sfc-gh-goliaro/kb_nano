"""Standardized workload definitions for eval (Tier 3).

These workloads are constants that ensure reproducible, comparable results
across runs and users. They are not configurable by design.

LLM workloads (text-only, random token IDs):
  Throughput: 3 scenarios (prefill-heavy, balanced, decode-heavy), 1000 reqs each.
  Latency: 2 scenarios (single-request, fixed-batch-32).

VLM workloads (multi-modal):
  Throughput: 3 scenarios (text-only, image, video), 1000 reqs each.
  Latency: 2 scenarios (single-image, single-video), batch_size=1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThroughputWorkload:
    name: str
    input_len: int
    output_len: int
    num_requests: int = 1000


@dataclass(frozen=True)
class LatencyWorkload:
    name: str
    batch_size: int
    input_len: int
    output_len: int
    num_warmup: int = 3
    num_iters: int = 5


THROUGHPUT_WORKLOADS: list[ThroughputWorkload] = [
    ThroughputWorkload(name="prefill-heavy", input_len=1024, output_len=512),
    ThroughputWorkload(name="balanced",      input_len=512,  output_len=512),
    ThroughputWorkload(name="decode-heavy",  input_len=512,  output_len=1024),
]

LATENCY_WORKLOADS: list[LatencyWorkload] = [
    LatencyWorkload(name="single-request",  batch_size=1,  input_len=128, output_len=128),
    LatencyWorkload(name="fixed-batch-32",  batch_size=32, input_len=128, output_len=128),
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
    input_len: int | None  # fixed input token length (text only)
    output_len: int
    dataset_name: str | None = None  # HF dataset (image/video only)
    num_requests: int = 1000


VLM_THROUGHPUT_WORKLOADS: list[VLMThroughputWorkload] = [
    VLMThroughputWorkload(
        "text-only", "text", input_len=512, output_len=1024),
    VLMThroughputWorkload(
        "image", "image", input_len=None, output_len=512,
        dataset_name="lmarena-ai/VisionArena-Chat"),
    VLMThroughputWorkload(
        "video", "video", input_len=None, output_len=512,
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


def get_max_seq_len() -> int:
    """Return the maximum sequence length across all standardized workloads."""
    max_len = 0
    for w in THROUGHPUT_WORKLOADS:
        max_len = max(max_len, w.input_len + w.output_len)
    for w in LATENCY_WORKLOADS:
        max_len = max(max_len, w.input_len + w.output_len)
    return max_len
