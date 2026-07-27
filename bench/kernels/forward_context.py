"""Minimal forward context + KV cache setup for Tier-1 kernel benchmarking.

Attention-bearing tasks (every L2 attention block, every L3 decoder layer, every
L4 pipeline) do not run standalone: they read paged-KV metadata from the global
``infra.context`` singleton and expect the engine to have attached real
``k_cache`` / ``v_cache`` tensors to each attention submodule
(``infra/engine.py:allocate_kv_cache``).  Constructed directly, those modules
carry ``self.k_cache = self.v_cache = torch.tensor([])`` and their forward fails
with ``cu_seqlens_k or seqused_k must be provided``.

The Tier-1 runner has no engine, so this module supplies the smallest context
that satisfies the production contract: the scenario is presented as a single
sequence in **prefill**, with contiguous slot mapping over freshly allocated
cache blocks.  That matches how the traced scenarios were produced (each
scenario is one forward of one batch) and keeps the comparison
baseline-vs-candidate symmetric, which is what correctness checking needs.

Only enough blocks for the scenario are allocated, so memory stays proportional
to the benchmark rather than to a full serving deployment.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn

_KV_HEAD_ATTRS = ("num_kv_heads", "num_key_value_heads", "n_kv_heads")
_HEAD_DIM_ATTRS = ("head_dim", "head_size", "hidden_size_per_head")


def _peer_context_modules(modules: list[nn.Module]) -> list[Any]:
    """Task modules that carry their own inlined copy of ``infra.context``.

    A reference implementation is required to be self-contained, and upstream
    satisfies that by inlining every dependency -- including ``infra/context.py``
    (``_CONTEXT = Context()`` at ``tasks/reference/L3/llama_decoder.py:752``).
    That module is not a kernel, though: it is the interface the *engine* uses to
    hand paged-KV metadata to the kernel.  Inlining it gives the reference a
    private singleton, so ``set_context`` on ``fastkernels.infra.context`` leaves
    the reference's own copy empty and every paged-attention reference dies on
    ``slot_mapping >= 0`` with ``'>=' not supported between NoneType and int``.

    Rather than edit the shipped references, mirror the context into each private
    copy so upstream's code runs exactly as published.
    """
    import sys

    seen: set[str] = set()
    peers = []
    for mod in modules:
        for cls in type(mod).__mro__:
            name = getattr(cls, "__module__", None)
            if not name or name in seen or name == "fastkernels.infra.context":
                continue
            seen.add(name)
            pymod = sys.modules.get(name)
            if pymod is None:
                continue
            if all(callable(getattr(pymod, fn, None))
                   for fn in ("set_context", "reset_context")):
                peers.append(pymod)
    return peers


def _attention_submodules(module: nn.Module) -> list[nn.Module]:
    """Submodules the engine would have attached a KV cache to."""
    found = []
    for sub in module.modules():
        k = getattr(sub, "k_cache", None)
        if isinstance(k, torch.Tensor):
            found.append(sub)
    return found


def _int_attr(mod: nn.Module, names: tuple[str, ...]) -> int | None:
    for name in names:
        val = getattr(mod, name, None)
        if isinstance(val, int) and val > 0:
            return val
    return None


def _num_tokens(inputs: Any) -> int:
    """Token count for the scenario.

    Layout differs per module family: a flat ``[tokens, hidden]`` activation
    puts tokens on dim 0, while a batched ``[batch, seq, hidden]`` one (GLA and
    the other linear-attention decoders) puts them on dim 1.  Taking dim 0
    unconditionally reports 1 token for a ``(1, 16360, 2560)`` input, and the
    resulting ``cu_seqlens=[0, 1]`` makes FLA's ``prepare_chunk_indices``
    compute a negative chunk count ("upper bound and lower bound inconsistent
    with step sign") on 31 of gla_decoder's 320 scenarios.

    For a 3-D activation the token count is batch*seq, which is what the packed
    varlen contract the context advertises actually means.
    """
    def _tokens_of(t: torch.Tensor) -> int:
        if t.ndim >= 3:
            return int(t.shape[0]) * int(t.shape[1])
        return int(t.shape[0])

    if isinstance(inputs, dict):
        for key in ("hidden_states", "x", "query", "input_ids", "positions"):
            val = inputs.get(key)
            if isinstance(val, torch.Tensor) and val.ndim >= 1:
                return _tokens_of(val)
        for val in inputs.values():
            if isinstance(val, torch.Tensor) and val.ndim >= 1:
                return _tokens_of(val)
    if isinstance(inputs, torch.Tensor) and inputs.ndim >= 1:
        return _tokens_of(inputs)
    return 1


def _sequence_lengths(inputs: Any) -> list[int]:
    """Per-row sequence lengths for a batched activation, else []."""
    def _of(t: torch.Tensor) -> list[int]:
        if t.ndim >= 3:
            return [int(t.shape[1])] * int(t.shape[0])
        return []
    if isinstance(inputs, dict):
        for key in ("hidden_states", "x", "query", "inputs_embeds"):
            v = inputs.get(key)
            if isinstance(v, torch.Tensor):
                r = _of(v)
                if r:
                    return r
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                r = _of(v)
                if r:
                    return r
    if isinstance(inputs, torch.Tensor):
        return _of(inputs)
    return []


@contextmanager
def tier1_forward_context(
    modules: list[nn.Module],
    inputs: Any,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> Iterator[bool]:
    """Attach KV caches to ``modules`` and install a single-sequence context.

    Yields True when a context was actually needed and installed.  All modules
    are set up together and share the same metadata so that a baseline and a
    candidate see byte-identical cache state.
    """
    from fastkernels.infra.context import (
        get_attn_backend_config,
        reset_context,
        set_context,
    )

    attn_mods: list[nn.Module] = []
    for mod in modules:
        attn_mods.extend(_attention_submodules(mod))

    if not attn_mods:
        yield False
        return

    cfg = get_attn_backend_config()
    block_size = int(getattr(cfg, "block_size", 256) or 256)
    hnd = getattr(cfg, "kv_layout", "NHD") == "HND"

    num_tokens = _num_tokens(inputs)
    num_blocks = max(1, -(-num_tokens // block_size))
    cache_dtype = dtype or torch.bfloat16

    saved: list[tuple[nn.Module, torch.Tensor, torch.Tensor]] = []
    for mod in attn_mods:
        saved.append((mod, mod.k_cache, mod.v_cache))
        n_kv = _int_attr(mod, _KV_HEAD_ATTRS) or 1
        hd = _int_attr(mod, _HEAD_DIM_ATTRS)
        if hd is None:
            # Fall back to the shape the module's own projection implies.
            hd = 128
        shape = (
            (num_blocks, n_kv, block_size, hd) if hnd
            else (num_blocks, block_size, n_kv, hd)
        )
        mod.k_cache = torch.zeros(shape, dtype=cache_dtype, device=device)
        mod.v_cache = torch.zeros(shape, dtype=cache_dtype, device=device)

    # cu_seqlens must describe *real* sequence boundaries.  Presenting a
    # [batch, seq, hidden] activation as one packed sequence of batch*seq
    # tokens makes FLA's prepare_chunk_indices derive a chunk count from a
    # length that no single sequence has, and 31 of gla_decoder's 320
    # scenarios die on "upper bound and lower bound inconsistent with step
    # sign".  Emit one boundary per batch row instead.
    seq_lens = _sequence_lengths(inputs)
    if seq_lens and len(seq_lens) > 1:
        bounds = [0]
        for n in seq_lens:
            bounds.append(bounds[-1] + n)
        cu = torch.tensor(bounds, dtype=torch.int32, device=device)
        max_seq = max(seq_lens)
    else:
        cu = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
        max_seq = num_tokens
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device=device,
    ).unsqueeze(0)
    context_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)

    ctx_kwargs = dict(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_seq,
        max_seqlen_k=max_seq,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        max_context_len=num_tokens,
    )
    peers = _peer_context_modules(modules + attn_mods)

    try:
        set_context(True, **ctx_kwargs)
        for pymod in peers:
            try:
                pymod.set_context(True, **ctx_kwargs)
            except TypeError:
                # An inlined copy whose signature drifted from the engine's:
                # pass only what it accepts rather than failing the scenario.
                import inspect
                try:
                    accepted = set(
                        inspect.signature(pymod.set_context).parameters
                    )
                except (TypeError, ValueError):
                    continue
                pymod.set_context(True, **{
                    k: v for k, v in ctx_kwargs.items() if k in accepted
                })
        yield True
    finally:
        reset_context()
        for pymod in peers:
            try:
                pymod.reset_context()
            except Exception:
                pass
        for mod, k, v in saved:
            mod.k_cache = k
            mod.v_cache = v
