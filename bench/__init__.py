"""
kb-nano benchmark suite for CUDA kernel evaluation.

Supports 4 abstraction levels:
  L1: Single-kernel ops (rms_norm, linear, rotary_emb, ...)
  L2: Multi-op blocks  (attention, llama_mlp, mixtral_moe, fused_experts)
  L3: Decoder layers   (llama_decoder, mixtral_decoder)
  L4: Full models      (llama, mixtral)

Usage:
    from kb_nano.bench import benchmark, list_targets, print_model_operator_map

    # See all available targets with their model mappings
    print_model_operator_map()

    # See all available targets
    for t in list_targets():
        print(f"  L{t.level} {t.name:30s} models={t.models}")

    # Benchmark a user kernel (must subclass the target's class)
    results = benchmark("rms_norm", MyRMSNorm)
"""

from .discovery import (
    BenchTarget,
    list_targets,
    models_for_target,
    print_model_operator_map,
    targets_for_model,
)
from .evaluator import BenchResult
from .runner import run_benchmark as benchmark

__all__ = [
    "benchmark",
    "list_targets",
    "models_for_target",
    "print_model_operator_map",
    "targets_for_model",
    "BenchResult",
    "BenchTarget",
]
