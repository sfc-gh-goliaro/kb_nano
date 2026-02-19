"""
KL divergence and speedup computation for benchmark evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class BenchResult:
    """Results from a single benchmark run (one target, one model)."""
    target_name: str
    model_name: str
    # KL divergence metrics (computed over per-step logits)
    kl_mean: float = 0.0
    kl_max: float = 0.0
    kl_per_step: list[float] = field(default_factory=list)
    # Timing
    baseline_time: float = 0.0
    user_time: float = 0.0
    speedup: float = 1.0
    # Token-level match rate
    token_match_rate: float = 1.0
    num_tokens: int = 0

    def report(self) -> str:
        lines = [
            f"Benchmark: {self.target_name} on {self.model_name}",
            f"  KL divergence:  mean={self.kl_mean:.6f}  max={self.kl_max:.6f}",
            f"  Token match:    {self.token_match_rate:.1%} ({self.num_tokens} tokens)",
            f"  Baseline time:  {self.baseline_time:.3f}s",
            f"  User time:      {self.user_time:.3f}s",
            f"  Speedup:        {self.speedup:.2f}x",
        ]
        return "\n".join(lines)


def compute_kl_divergence(
    baseline_logits: list[torch.Tensor],
    user_logits: list[torch.Tensor],
) -> tuple[float, float, list[float]]:
    """Compute per-step KL(p_baseline || p_user) and return (mean, max, per_step).

    Each entry in the lists is a [1, vocab_size] logit tensor for one decoding step.
    """
    if not baseline_logits or not user_logits:
        return 0.0, 0.0, []

    n = min(len(baseline_logits), len(user_logits))
    kl_values = []

    for i in range(n):
        bl = baseline_logits[i].float()
        ul = user_logits[i].float()
        # Ensure same device
        if bl.device != ul.device:
            ul = ul.to(bl.device)
        log_p = F.log_softmax(bl, dim=-1)
        log_q = F.log_softmax(ul, dim=-1)
        kl = F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
        kl_values.append(max(kl.item(), 0.0))

    kl_mean = sum(kl_values) / len(kl_values) if kl_values else 0.0
    kl_max = max(kl_values) if kl_values else 0.0
    return kl_mean, kl_max, kl_values


def compute_token_match_rate(
    baseline_ids: list[int],
    user_ids: list[int],
) -> tuple[float, int]:
    """Fraction of generated tokens that match exactly."""
    n = min(len(baseline_ids), len(user_ids))
    if n == 0:
        return 1.0, 0
    matches = sum(1 for a, b in zip(baseline_ids[:n], user_ids[:n]) if a == b)
    return matches / n, n


def evaluate(
    target_name: str,
    model_name: str,
    baseline_outputs,
    user_outputs,
    baseline_time: float,
    user_time: float,
) -> BenchResult:
    """Compare baseline and user outputs, producing a BenchResult."""
    all_kl_per_step = []
    total_match = 0
    total_tokens = 0

    for bo, uo in zip(baseline_outputs, user_outputs):
        if bo.logits_history and uo.logits_history:
            _, _, kl_steps = compute_kl_divergence(
                bo.logits_history, uo.logits_history,
            )
            all_kl_per_step.extend(kl_steps)

        rate, n = compute_token_match_rate(bo.token_ids, uo.token_ids)
        total_match += int(rate * n)
        total_tokens += n

    kl_mean = sum(all_kl_per_step) / len(all_kl_per_step) if all_kl_per_step else 0.0
    kl_max = max(all_kl_per_step) if all_kl_per_step else 0.0
    match_rate = total_match / total_tokens if total_tokens else 1.0
    speedup = baseline_time / user_time if user_time > 0 else float("inf")

    return BenchResult(
        target_name=target_name,
        model_name=model_name,
        kl_mean=kl_mean,
        kl_max=kl_max,
        kl_per_step=all_kl_per_step,
        baseline_time=baseline_time,
        user_time=user_time,
        speedup=speedup,
        token_match_rate=match_rate,
        num_tokens=total_tokens,
    )
