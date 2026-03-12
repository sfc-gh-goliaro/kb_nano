"""Standardized workload definitions for eval (Tier 3).

These workloads are constants that ensure reproducible, comparable results
across runs and users. They are not configurable by design.

Throughput workloads: 3 scenarios covering prefill-heavy, balanced, and
decode-heavy request mixes, each with 1000 requests.

Latency workloads: 2 scenarios covering single-request and fixed-batch-32
inference latency.
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


def get_max_seq_len() -> int:
    """Return the maximum sequence length across all standardized workloads."""
    max_len = 0
    for w in THROUGHPUT_WORKLOADS:
        max_len = max(max_len, w.input_len + w.output_len)
    for w in LATENCY_WORKLOADS:
        max_len = max(max_len, w.input_len + w.output_len)
    return max_len
