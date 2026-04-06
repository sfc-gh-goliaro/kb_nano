"""Custom op registrations for torch.compile boundaries.

Registers opaque custom ops for attention so that torch.compile (Inductor)
does not trace into paged-KV attention kernels.  At runtime, the ops look
up the actual nn.Module from the global ``no_compile_layers`` registry and
call its implementation.

Matching vLLM's default for Qwen3-VL-235B-FP8: splitting_ops contains only
attention ops.  MoE is NOT a splitting op — the MoE forward is transparent
to Inductor (it appears as opaque nodes within a compiled piece, not as a
graph boundary).  This lets Inductor optimize the code around MoE (norms,
linears) within the same compiled subgraph.

MoE custom ops are still registered (for use when MoE needs to be opaque,
e.g. expert parallelism), but they are not in SPLITTING_OPS by default.
"""

from __future__ import annotations

import torch

from .context import get_no_compile_layers

# Only attention ops are splitting points, matching vLLM's _attention_ops.
# MoE ops are NOT splitting points — they appear as opaque nodes within
# compiled subgraphs, not as boundaries between subgraphs.
SPLITTING_OPS: list[str] = [
    "kb_nano::unified_attention",
]


# ---------------------------------------------------------------------------
# MoE custom op
# ---------------------------------------------------------------------------

def _moe_forward_impl(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    return layer.forward_impl(hidden_states)


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Attention custom op
# ---------------------------------------------------------------------------

def _unified_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_no_compile_layers()[layer_name]
    return layer.forward_impl(query, key, value)


def _unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_registered = False


def ensure_custom_ops_registered() -> None:
    """Register the custom ops with torch.library (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True

    lib = torch.library.Library("kb_nano", "DEF")

    lib.define(
        "moe_forward(Tensor hidden_states, str layer_name) -> Tensor"
    )
    lib.impl("moe_forward", _moe_forward_impl, "CUDA")
    lib.impl("moe_forward", _moe_forward_impl, "CPU")

    abstract_lib = torch.library.Library("kb_nano", "IMPL", "Meta")
    abstract_lib.impl("moe_forward", _moe_forward_fake)

    lib.define(
        "unified_attention(Tensor query, Tensor key, Tensor value, "
        "str layer_name) -> Tensor"
    )
    lib.impl("unified_attention", _unified_attention_impl, "CUDA")
    lib.impl("unified_attention", _unified_attention_impl, "CPU")
    abstract_lib.impl("unified_attention", _unified_attention_fake)

    # Keep references alive for the lifetime of the process.
    ensure_custom_ops_registered._lib = lib  # type: ignore[attr-defined]
    ensure_custom_ops_registered._abstract_lib = abstract_lib  # type: ignore[attr-defined]
