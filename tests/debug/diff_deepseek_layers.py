"""Operator-by-operator output comparison between kb_nano and vLLM
for DeepSeek-V3.2.

Both engines load a *truncated* DeepSeek-V3.2 (only the first ``--num-layers``
decoder blocks; default 5) so the model fits on a single H200 with TP=1.
A symlinked staging directory is built on disk that mirrors the original
HF checkpoint but with ``num_hidden_layers`` patched in ``config.json``;
both engines silently drop the unused layers' weight tensors.

Usage
-----

    # Run kb-nano forward, dump activations to /tmp/acts_kb.pt
    python diff_deepseek_layers.py --engine kb_nano --output /tmp/acts_kb.pt

    # Run vLLM forward, dump activations to /tmp/acts_vllm.pt
    python diff_deepseek_layers.py --engine vllm     --output /tmp/acts_vllm.pt

    # Diff both dumps
    python diff_deepseek_layers.py --diff /tmp/acts_kb.pt /tmp/acts_vllm.pt

The two engines are run sequentially (not concurrent) — each takes the full
GPU. Because both processes serialize the *same* fixed-token prompt and we
only hook TP-replicated module boundaries, the dumps are directly
comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Make sure we import the *local* kb_nano source, not any installed copy.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch


_STEP_STATE: dict = {"counter": {"i": 0}}


def _step() -> int:
    return _STEP_STATE["counter"]["i"]


# ---------------------------------------------------------------------------
# Parallel sharded dump save / load
# ---------------------------------------------------------------------------
# ``torch.save`` of a dict with tens of thousands of small tensors is CPU /
# pickle-bound — we've measured ~40 MB/s on this box which makes a 50 GB
# dump take > 20 min.  Instead we split the dump into N shards and write
# each shard as a ``safetensors`` file in parallel from a thread pool.
# Safetensors is essentially a memcpy (no pickle), and the GIL is released
# for the actual write, so a 16-way thread pool saturates NVMe bandwidth.
#
# The output path is interpreted as a *directory* (we add ``.d`` if the
# caller passes something ending in ``.pt``).  Layout:
#   <dir>/metadata.json               engine name + total tensor count
#   <dir>/shard_<000..NNN>.safetensors
def _save_dump_parallel(
    dump: dict,
    output_path: str,
    engine: str,
    n_shards: int = 32,
) -> None:
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from safetensors.torch import save_file as _safe_save
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "safetensors is required for parallel dump save; "
            "pip install safetensors"
        ) from e
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(it, **kw):  # type: ignore
            return it

    out_dir = Path(output_path)
    if out_dir.suffix == ".pt":
        out_dir = Path(str(output_path) + ".d")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Round-robin assignment of tensors to shards keeps shard sizes even
    # regardless of tensor-size distribution.
    items = list(dump.items())
    shards: list[list[tuple[str, torch.Tensor]]] = [
        [] for _ in range(n_shards)
    ]
    for i, kv in enumerate(items):
        shards[i % n_shards].append(kv)

    def _save_shard(idx: int, shard_items):
        # ``safetensors`` requires contiguous tensors and refuses to save
        # tensors that share storage -- clone to be safe.
        d = {}
        for k, v in shard_items:
            if isinstance(v, torch.Tensor):
                if not v.is_contiguous():
                    v = v.contiguous()
                # ``clone`` costs a copy but the dump tensors are CPU-side
                # float32/int tensors typically < 1MB each, so this is
                # well worth avoiding safetensors storage-sharing errors.
                d[k] = v.clone()
            else:
                d[k] = v
        path = out_dir / f"shard_{idx:03d}.safetensors"
        _safe_save(d, str(path))
        return len(d)

    print(f"[{engine}] saving {len(items)} tensors across {n_shards} "
          f"safetensors shards to {out_dir} ...", flush=True)
    total = 0
    with ThreadPoolExecutor(max_workers=n_shards) as ex:
        futs = {
            ex.submit(_save_shard, i, s): i
            for i, s in enumerate(shards)
        }
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"[{engine}] shards",
            dynamic_ncols=True,
        ):
            total += fut.result()

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "engine": engine,
            "num_tensors": total,
            "num_shards": n_shards,
        }, f)
    print(f"[{engine}] saved {total} tensors to {out_dir}", flush=True)


def _load_dump_parallel(path: str) -> tuple[str, dict]:
    """Load a dump previously written by :func:`_save_dump_parallel`.

    Also transparently falls back to the legacy single-file ``torch.save``
    format when the path points to a ``.pt`` file that exists directly
    (rather than its ``.d`` directory sibling).
    """
    import json
    from concurrent.futures import ThreadPoolExecutor

    try:
        from safetensors.torch import load_file as _safe_load
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("safetensors is required to load dumps") from e
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(it, **kw):  # type: ignore
            return it

    out_dir = Path(path)
    # Legacy ``torch.save`` artefact?
    if out_dir.is_file() and out_dir.suffix == ".pt":
        blob = torch.load(str(out_dir), map_location="cpu", weights_only=False)
        return blob["engine"], blob["tensors"]
    if out_dir.suffix == ".pt":
        out_dir = Path(str(path) + ".d")
    if not out_dir.is_dir():
        raise FileNotFoundError(f"no dump found at {path} (or {out_dir})")
    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)
    engine = meta["engine"]
    shard_paths = sorted(out_dir.glob("shard_*.safetensors"))
    tensors: dict = {}

    def _load_one(sp):
        return _safe_load(str(sp))

    with ThreadPoolExecutor(max_workers=min(16, len(shard_paths))) as ex:
        for shard in tqdm(
            ex.map(_load_one, shard_paths),
            total=len(shard_paths),
            desc=f"[{engine}] load",
            dynamic_ncols=True,
        ):
            tensors.update(shard)
    return engine, tensors


# Fixed prompt token IDs.  These are arbitrary but deterministic — both
# engines see the same input.  We use a short prefill so the activation
# tensors stay small and the run completes quickly.
PROMPT_TOKEN_IDS = [
    1, 9, 14, 17, 23, 31, 42, 53, 71, 89,
    101, 113, 127, 137, 149, 163,
]
SEED = 0xC0FFEE


# ---------------------------------------------------------------------------
# Staging dir: mirror HF checkpoint with num_hidden_layers patched
# ---------------------------------------------------------------------------
def make_truncated_checkpoint_dir(
    model_name: str,
    num_layers: int,
    base_cache_dir: str = "/tmp/kbnano_diff_checkpoints",
) -> str:
    """Materialise a directory that looks like the HF snapshot of
    ``model_name`` but with:

      * ``num_hidden_layers`` patched in ``config.json`` to ``num_layers``
      * Only the safetensors files that contain weights for layers
        ``[0, num_layers)`` (or non-layer-specific weights such as the
        embedding / final norm / lm_head) are *included*; others are
        omitted entirely so vLLM doesn't see "extra" tensors that don't
        match any model parameter.

    The included safetensors are *rewritten* (not symlinked) with only
    the relevant tensors so vLLM never sees a stray ``model.layers.7.*``
    weight.
    """
    import re
    import safetensors.torch as st
    from huggingface_hub import snapshot_download

    src = snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
    )
    dst = Path(base_cache_dir) / f"{model_name.replace('/', '__')}__L{num_layers}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Pattern: model.layers.<idx>.* -> capture idx
    layer_re = re.compile(r"model\.layers\.(\d+)\.")

    def _keep_key(k: str) -> bool:
        m = layer_re.search(k)
        if m is None:
            return True   # embeddings / norm / lm_head etc.
        return int(m.group(1)) < num_layers

    # Walk the source, copy json files (patching config.json), and rewrite
    # safetensors with only the kept tensors.  Skip .safetensors files that
    # contribute *no* tensors after filtering so vLLM doesn't try to load
    # an empty file.
    n_files_kept = 0
    for entry in sorted(os.listdir(src)):
        sp = os.path.join(src, entry)
        dp = os.path.join(dst, entry)
        if entry == "config.json":
            with open(sp) as f:
                cfg = json.load(f)
            cfg["num_hidden_layers"] = num_layers
            # Patch first_k_dense_replace as well: we need at least 1
            # MoE layer in our slice.  Default in V3.2 = 3 (so layer 3
            # is the first MoE layer).  Don't touch otherwise.
            with open(dp, "w") as f:
                json.dump(cfg, f, indent=2)
        elif entry.endswith(".safetensors"):
            # Inspect tensor names without loading data
            from safetensors import safe_open
            with safe_open(sp, framework="pt") as f:
                keys = list(f.keys())
            kept_keys = [k for k in keys if _keep_key(k)]
            if not kept_keys:
                continue
            if len(kept_keys) == len(keys):
                # All tensors useful - just symlink the original file.
                os.symlink(sp, dp)
            else:
                # Mixed - need to materialise a filtered copy.  Load only
                # the kept tensors to keep memory low.
                with safe_open(sp, framework="pt") as f:
                    kept = {k: f.get_tensor(k) for k in kept_keys}
                st.save_file(kept, dp)
            n_files_kept += 1
        elif entry.endswith(".safetensors.index.json"):
            # Filter the index too, in case there is one
            with open(sp) as f:
                idx = json.load(f)
            wm = idx.get("weight_map", {})
            wm = {k: v for k, v in wm.items() if _keep_key(k)}
            idx["weight_map"] = wm
            with open(dp, "w") as f:
                json.dump(idx, f)
        else:
            os.symlink(sp, dp)

    print(f"[stage] truncated checkpoint at {dst} "
          f"(num_hidden_layers={num_layers}, "
          f"safetensors_files={n_files_kept})", flush=True)
    return str(dst)


def _install_kb_nano_sparse_fwd_hook(dump: dict, handles: list) -> None:
    """Patch ``flash_mla_sparse_fwd`` (vendored under
    ``vllm.third_party.flashmla``) to capture every call's inputs and
    outputs. Also patches every already-imported re-binding of the
    symbol in vLLM and kb_nano modules. Defined at module scope so that
    its dotted-name imports (``third_party``, ``flashmla``,
    ``backends``, etc.) do not appear in :func:`install_layer_hooks`'s
    ``co_names`` and therefore do not get pulled into the cloudpickle
    submodule chase that triggers the ``GenericModule`` failure.
    """
    try:
        import importlib
        _fmi = importlib.import_module(
            "vllm.third_party.flashmla.flash_mla_interface"
        )
        _orig_sparse_fwd = _fmi.flash_mla_sparse_fwd
        _sparse_call_idx = {"i": 0}

        def _wrapped_sparse_fwd(q, kv, indices, sm_scale, *args, **kwargs):
            i = _sparse_call_idx["i"]
            _sparse_call_idx["i"] += 1
            tag = f"flash_mla_sparse_fwd_call{i}"
            # Don't print on every call when running multi-token decode --
            # 61 layers * many steps is too noisy.  Print only the first
            # few of each step.
            if i < 3 or i % 61 == 0:
                print(
                    f"[diag] >>> flash_mla_sparse_fwd call {i}: "
                    f"q={tuple(q.shape)} kv={tuple(kv.shape)} "
                    f"indices={tuple(indices.shape)}",
                    flush=True,
                )
            try:
                dump[f"{tag}.q"] = q.detach().to(torch.float32).cpu()
                # Always skip the KV cache dump -- it's the paged buffer
                # which is the same per call (just grows over time) and
                # dominates dump size.  The attention comparison only
                # needs q / indices / output for divergence detection.
                dump[f"{tag}.kv_skipped"] = torch.tensor(
                    [kv.numel(), int(kv.element_size())], dtype=torch.int64,
                )
                dump[f"{tag}.indices"] = indices.detach().to(torch.int64).cpu()
                dump[f"{tag}.sm_scale"] = torch.tensor(float(sm_scale))
                dump[f"{tag}.q_shape"] = torch.tensor(list(q.shape))
                dump[f"{tag}.kv_shape"] = torch.tensor(list(kv.shape))
                dump[f"{tag}.indices_shape"] = torch.tensor(list(indices.shape))
            except Exception as e:
                print(f"[diag] failed to snapshot sparse_fwd input: {e}")
            out = _orig_sparse_fwd(q, kv, indices, sm_scale, *args, **kwargs)
            try:
                if isinstance(out, tuple):
                    o = out[0]
                else:
                    o = out
                if isinstance(o, torch.Tensor):
                    dump[f"{tag}.out"] = o.detach().to(torch.float32).cpu()
            except Exception as e:
                print(f"[diag] failed to snapshot sparse_fwd output: {e}")
            return out

        _fmi.flash_mla_sparse_fwd = _wrapped_sparse_fwd
        rebind_modules = []
        for modname in (
            "vllm.v1.attention.backends.mla.flashmla_sparse",
            "kb_nano.tasks.baseline.L1._flashmla_backend",
            "kb_nano.tasks.baseline.L1.flash_mla_sparse_prefill",
        ):
            try:
                m = importlib.import_module(modname)
                if hasattr(m, "flash_mla_sparse_fwd"):
                    m.flash_mla_sparse_fwd = _wrapped_sparse_fwd
                    rebind_modules.append(m)
            except Exception:
                pass

        class _SparsePatchRestorer:
            def remove(self):
                _fmi.flash_mla_sparse_fwd = _orig_sparse_fwd
                for m in rebind_modules:
                    try:
                        m.flash_mla_sparse_fwd = _orig_sparse_fwd
                    except Exception:
                        pass

        handles.append(_SparsePatchRestorer())
        print("[diag] installed flash_mla_sparse_fwd input/output hook",
              flush=True)
    except Exception as e:
        print(f"[diag] could not install flash_mla_sparse_fwd hook: {e}",
              flush=True)


def _install_grouped_topk_hook(dump: dict, handles: list) -> None:
    """Patch ``vllm._custom_ops.grouped_topk`` to capture inputs and
    outputs for each call. Dumps under ``grouped_topk.call{i}.*`` so we
    can diff the kernel's inputs/outputs across engines.
    """
    try:
        import importlib
        _vops = importlib.import_module("vllm._custom_ops")
        _orig = _vops.grouped_topk
        _idx = {"i": 0}

        def _wrapped(*args, **kwargs):
            i = _idx["i"]
            _idx["i"] += 1
            tag = f"grouped_topk.call{i}"
            try:
                pos = args if args else None
                if pos and isinstance(pos[0], torch.Tensor):
                    dump[f"{tag}.gating_output"] = (
                        pos[0].detach().to(torch.float32).cpu()
                    )
                if len(pos) >= 7 and isinstance(pos[6], torch.Tensor):
                    dump[f"{tag}.bias"] = (
                        pos[6].detach().to(torch.float32).cpu()
                    )
                # Scalar metadata for debugging.
                dump[f"{tag}.meta"] = torch.tensor(
                    [pos[j] if (j < len(pos) and isinstance(pos[j], (int, float, bool)))
                     else -1 for j in (1, 2, 3, 4, 5, 7)],
                    dtype=torch.float64,
                )
            except Exception as e:
                print(f"[diag] grouped_topk input snap failed: {e}",
                      flush=True)
            out = _orig(*args, **kwargs)
            try:
                if isinstance(out, tuple) and len(out) == 2:
                    dump[f"{tag}.topk_weights"] = (
                        out[0].detach().to(torch.float32).cpu()
                    )
                    dump[f"{tag}.topk_ids"] = (
                        out[1].detach().to(torch.int64).cpu()
                    )
            except Exception as e:
                print(f"[diag] grouped_topk output snap failed: {e}",
                      flush=True)
            return out

        _vops.grouped_topk = _wrapped

        # kb_nano's grouped_topk module caches the function reference once,
        # clear that cache so our wrapper is picked up.
        try:
            _gt = importlib.import_module(
                "kb_nano.tasks.baseline.L1.grouped_topk"
            )
            _gt._FUSED_GROUPED_TOPK = None
            _gt._FUSED_GROUPED_TOPK_RESOLVED = False
        except Exception:
            pass

        class _R:
            def remove(self):
                _vops.grouped_topk = _orig

        handles.append(_R())
        print("[diag] installed grouped_topk input/output hook",
              flush=True)
    except Exception as e:
        print(f"[diag] could not install grouped_topk hook: {e}",
              flush=True)


def _install_deepgemm_grouped_hook(dump: dict, handles: list) -> None:
    """Patch ``deep_gemm.m_grouped_fp8_gemm_nt_contiguous`` to capture
    the inputs and outputs of every routed-experts FP8 GEMM call. Both
    kb_nano (via ``Fp8GroupedGemmContiguous``) and vLLM (via
    ``vllm.utils.deep_gemm.m_grouped_fp8_gemm_nt_contiguous``) ultimately
    dispatch into the same ``deep_gemm`` symbol, so capturing it once
    here covers both engines. We capture only the first few calls to
    keep the dump size bounded — there are 2 GEMM calls per MoE layer
    (w13 then w2).
    """
    try:
        import importlib
        _dg = importlib.import_module("deep_gemm")
        _orig = _dg.m_grouped_fp8_gemm_nt_contiguous
        _idx = {"i": 0}
        max_calls = 24  # 2 GEMMs * up to 12 MoE layers

        def _wrapped(a, b, c, expert_ids, *args, **kwargs):
            i = _idx["i"]
            _idx["i"] += 1
            print(f"[diag] >>> deepgemm m_grouped call#{i}", flush=True)
            if i < max_calls:
                tag = f"deepgemm.call{i}"
                try:
                    a_fp8, a_scale = a
                    b_fp8, b_scale = b
                    dump[f"{tag}.a_fp8"] = a_fp8.detach().cpu().to(torch.uint8)
                    dump[f"{tag}.a_scale"] = a_scale.detach().to(torch.float32).cpu()
                    dump[f"{tag}.expert_ids"] = expert_ids.detach().to(torch.int64).cpu()
                    dump[f"{tag}.a_shape"] = torch.tensor(list(a_fp8.shape))
                    dump[f"{tag}.b_shape"] = torch.tensor(list(b_fp8.shape))
                    dump[f"{tag}.c_shape"] = torch.tensor(list(c.shape))
                    dump[f"{tag}.a_strides"] = torch.tensor(list(a_fp8.stride()))
                    dump[f"{tag}.a_scale_strides"] = torch.tensor(
                        list(a_scale.stride()))
                    # Capture small chunk of weight FP8 to verify they
                    # are bit-identical across engines.
                    dump[f"{tag}.b_fp8_first_block"] = (
                        b_fp8.flatten()[:128].detach().cpu().to(torch.uint8)
                    )
                    dump[f"{tag}.b_scale_first_block"] = (
                        b_scale.flatten()[:128].detach().to(torch.float32).cpu()
                    )
                except Exception as e:
                    print(f"[diag] deepgemm input snap failed: {e}",
                          flush=True)
            ret = _orig(a, b, c, expert_ids, *args, **kwargs)
            if i < max_calls:
                try:
                    dump[f"deepgemm.call{i}.c_out"] = (
                        c.detach().to(torch.float32).cpu()
                    )
                except Exception as e:
                    print(f"[diag] deepgemm output snap failed: {e}",
                          flush=True)
            return ret

        _dg.m_grouped_fp8_gemm_nt_contiguous = _wrapped

        # Trace which DeepGEMM-style experts class actually fires.
        try:
            from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
                DeepGemmExperts,
            )
            _orig_apply = DeepGemmExperts.apply
            _ap_calls = {"i": 0}
            def _trace_apply(self, *a, **kw):
                print(f"[diag] >>> DeepGemmExperts.apply call#{_ap_calls['i']}",
                      flush=True)
                _ap_calls["i"] += 1
                return _orig_apply(self, *a, **kw)
            DeepGemmExperts.apply = _trace_apply
        except Exception as e:
            print(f"[diag] could not trace DeepGemmExperts.apply: {e}",
                  flush=True)
        try:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                TritonExperts,
            )
            _orig_te_apply = TritonExperts.apply
            _te_calls = {"i": 0}
            def _trace_te(self, *a, **kw):
                print(f"[diag] >>> TritonExperts.apply call#{_te_calls['i']}",
                      flush=True)
                _te_calls["i"] += 1
                return _orig_te_apply(self, *a, **kw)
            TritonExperts.apply = _trace_te
        except Exception as e:
            print(f"[diag] could not trace TritonExperts.apply: {e}",
                  flush=True)

        # vLLM caches the resolved symbol via ``_lazy_init`` into
        # ``_grouped_impl``; patch that as well, plus the public wrapper
        # symbol re-exported by the modules below. kb_nano binds it
        # directly via ``deep_gemm.m_grouped_fp8_gemm_nt_contiguous`` so
        # the top-level patch already handles it.
        try:
            _vdg = importlib.import_module("vllm.utils.deep_gemm")
            _vdg._lazy_init()  # ensure the impl is resolved
            # Replace the cached _grouped_impl so vLLM's
            # ``m_grouped_fp8_gemm_nt_contiguous`` wrapper dispatches
            # into our snapshot wrapper.
            _vdg._grouped_impl = _wrapped
            # Also replace the public wrapper directly so consumers that
            # imported it by name (e.g. ``deep_gemm_moe.py``) pick it up.
            def _vdg_wrapped(*a, **kw):
                kw.pop("disable_ue8m0_cast", None)
                return _wrapped(*a, **kw)
            _vdg.m_grouped_fp8_gemm_nt_contiguous = _vdg_wrapped
            print("[diag] patched vllm.utils.deep_gemm._grouped_impl",
                  flush=True)
        except Exception as e:
            print(f"[diag] failed to patch vllm.utils.deep_gemm: {e}",
                  flush=True)

        # Rebind any consumer modules that already imported the symbol.
        for modname in (
            "vllm.model_executor.layers.fused_moe.deep_gemm_moe",
            "kb_nano.tasks.baseline.L1.fp8_grouped_gemm_contiguous",
        ):
            try:
                m = importlib.import_module(modname)
                if hasattr(m, "m_grouped_fp8_gemm_nt_contiguous"):
                    m.m_grouped_fp8_gemm_nt_contiguous = _wrapped
            except Exception:
                pass

        class _R:
            def remove(self):
                _dg.m_grouped_fp8_gemm_nt_contiguous = _orig

        handles.append(_R())
        print("[diag] installed deep_gemm m_grouped_fp8_gemm_nt_contiguous "
              "input/output hook", flush=True)
    except Exception as e:
        print(f"[diag] could not install deepgemm grouped hook: {e}",
              flush=True)


def _install_dsv3_router_gemm_hook(dump: dict, handles: list) -> None:
    """Patch ``vllm._custom_ops.dsv3_router_gemm`` to capture inputs and
    outputs for the first N calls. Dumps ``gate.call{i}.hidden_in``,
    ``gate.call{i}.gate_weight``, ``gate.call{i}.output`` to ``dump``.

    Both kb_nano and vLLM dispatch their MoE router gate through this
    kernel (DSV3 specialised path, SM90+, batch<=16, 256 experts,
    hidden=7168). Comparing its inputs directly lets us rule out any
    subtle input-normalisation difference between the two engines.
    """
    try:
        import importlib
        _vops = importlib.import_module("vllm._custom_ops")
        _orig = _vops.dsv3_router_gemm
        _idx = {"i": 0}

        def _wrapped(hidden_states=None, router_weight=None,
                     output_dtype=None, **kwargs):
            if hidden_states is None:
                hidden_states = kwargs.get("hidden_states")
            if router_weight is None:
                router_weight = kwargs.get("router_weight")
            if output_dtype is None:
                output_dtype = kwargs.get("output_dtype")
            i = _idx["i"]
            _idx["i"] += 1
            tag = f"gate.call{i}"
            try:
                dump[f"{tag}.hidden_in"] = (
                    hidden_states.detach().to(torch.float32).cpu()
                )
                dump[f"{tag}.gate_weight"] = (
                    router_weight.detach().to(torch.float32).cpu()
                )
                print(
                    f"[diag] dsv3_router_gemm call{i}: "
                    f"x.dtype={hidden_states.dtype} "
                    f"w.dtype={router_weight.dtype} "
                    f"out_dtype={output_dtype}",
                    flush=True,
                )
            except Exception as e:
                print(f"[diag] dsv3_router_gemm input snap failed: {e}",
                      flush=True)
            out = _orig(
                hidden_states=hidden_states,
                router_weight=router_weight,
                output_dtype=output_dtype,
            )
            try:
                dump[f"{tag}.output"] = out.detach().to(torch.float32).cpu()
            except Exception as e:
                print(f"[diag] dsv3_router_gemm output snap failed: {e}",
                      flush=True)
            return out

        _vops.dsv3_router_gemm = _wrapped

        # kb_nano's gate_linear caches the kernel handle via functools.cache
        # at first call. If warmup has already populated the cache, our
        # patch of _vops.dsv3_router_gemm is ignored. Clear the cache so
        # the wrapper is picked up on the next call.
        try:
            _gl = importlib.import_module(
                "kb_nano.tasks.baseline.L1.gate_linear"
            )
            if hasattr(_gl, "_maybe_load_router_kernels"):
                try:
                    _gl._maybe_load_router_kernels.cache_clear()
                except Exception:
                    pass
        except Exception:
            pass

        class _R:
            def remove(self):
                _vops.dsv3_router_gemm = _orig

        handles.append(_R())
        print("[diag] installed dsv3_router_gemm input/output hook",
              flush=True)
    except Exception as e:
        print(f"[diag] could not install dsv3_router_gemm hook: {e}",
              flush=True)


def _install_kb_nano_mla_dispatch_traces() -> None:
    """Trace kb_nano's ``MLAAttention._forward_sparse_separate`` and
    ``_forward_mha`` to confirm which dispatch path each batch takes.
    Module-scope to keep the kb_nano import out of
    :func:`install_layer_hooks`'s ``co_names``.
    """
    try:
        import importlib
        _mlai = importlib.import_module(
            "kb_nano.tasks.baseline.L2.mla_attention_impl"
        )
        _orig_sep = _mlai.MLAAttention._forward_sparse_separate
        _sep_calls = {"i": 0}

        def _trace_sep(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx,
                       topk_indices, num_prefill_tokens, num_decode_tokens):
            i = _sep_calls["i"]
            _sep_calls["i"] += 1
            bt_shape = (
                tuple(ctx.block_tables.shape) if ctx.block_tables is not None
                else "None"
            )
            tk_shape = (
                tuple(topk_indices.shape) if topk_indices is not None
                else "None"
            )
            print(
                f"[diag] MLA._forward_sparse_separate call#{i}: "
                f"block_tables={bt_shape} topk_indices={tk_shape} "
                f"is_prefill={ctx.is_prefill} is_mixed={ctx.is_mixed} "
                f"num_pf={num_prefill_tokens} num_dc={num_decode_tokens}",
                flush=True,
            )
            return _orig_sep(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache,
                             ctx, topk_indices, num_prefill_tokens,
                             num_decode_tokens)

        _mlai.MLAAttention._forward_sparse_separate = _trace_sep

        _orig_mha = _mlai.MLAAttention._forward_mha
        _mha_calls = {"i": 0}

        def _trace_mha(self, *a, **kw):
            i = _mha_calls["i"]
            _mha_calls["i"] += 1
            print(f"[diag] MLA._forward_mha call#{i} (DENSE PREFILL)",
                  flush=True)
            return _orig_mha(self, *a, **kw)

        _mlai.MLAAttention._forward_mha = _trace_mha
        print("[diag] installed MLA dispatch traces", flush=True)
    except Exception as e:
        print(f"[diag] could not install MLA dispatch traces: {e}",
              flush=True)


def _install_vllm_mla_dispatch_traces() -> None:
    """Install dispatch tracers on vLLM's FlashMLASparseImpl methods.

    Extracted into its own helper so the symbol ``backends`` (from
    ``vllm.v1.attention.backends.mla``) does not appear in the
    ``co_names`` of :func:`install_layer_hooks`. Otherwise cloudpickle's
    ``_find_imported_submodules`` will pull ``torch.backends`` (a
    ``GenericModule`` that cannot be pickled) into the function's
    submodule set when the worker function is serialized for the vLLM
    EngineCore process.
    """
    try:
        import importlib
        _vfms = importlib.import_module(
            "vllm.v1.attention.backends.mla.flashmla_sparse"
        )
        for _meth in (
            "_forward_bf16_kv",
            "_forward_fp8_kv_separate_prefill_decode",
            "_forward_fp8_kv_mixed_batch",
            "_bf16_flash_mla_kernel",
            "_fp8_flash_mla_kernel",
            "forward_mqa",
        ):
            _orig = getattr(_vfms.FlashMLASparseImpl, _meth, None)
            if _orig is None:
                continue

            def _make_trace_v(orig, name):
                _calls = {"i": 0}

                def _t(self, *a, **kw):
                    print(f"[diag] vLLM.MLA.{name} call#{_calls['i']}",
                          flush=True)
                    _calls["i"] += 1
                    return orig(self, *a, **kw)
                return _t

            setattr(_vfms.FlashMLASparseImpl, _meth,
                    _make_trace_v(_orig, _meth))
        print("[diag] installed vLLM MLA dispatch traces", flush=True)
    except Exception as e:
        print(f"[diag] could not install vLLM MLA dispatch traces: {e}",
              flush=True)


# ---------------------------------------------------------------------------
# Hook installation: register forward hooks on every interesting module
# ---------------------------------------------------------------------------
def install_layer_hooks(model, dump: dict, num_layers: int,
                        prefix_path: list[str],
                        engine: str = "kb_nano") -> list:
    import torch.nn as nn
    """Walk ``model`` until we find ``model.embed_tokens``, ``model.layers``,
    ``model.norm`` (the names match between kb_nano and vLLM).  Install
    forward hooks on each interesting submodule of layers ``[0, num_layers)``.

    Also monkey-patches the MoE module's ``forward``/``forward_impl`` so
    we can capture pre-allreduce, pre-scale intermediates that aren't
    surfaced through any submodule (e.g. router logits, top-k weights/ids,
    routed-experts output, shared-experts output).

    Returns a list of hook handles so the caller can remove them.
    """
    # Find the inner DeepseekV2Model / DeepSeekV3Model (both engines wrap it
    # under ``ForCausalLM.model``).
    inner = model
    for attr in prefix_path:
        if not hasattr(inner, attr):
            break
        inner = getattr(inner, attr)
    if not hasattr(inner, "layers"):
        # Try to walk children to find the one with .layers
        for child in model.modules():
            if hasattr(child, "layers") and hasattr(child, "embed_tokens"):
                inner = child
                break
    assert hasattr(inner, "layers"), \
        f"Could not find inner model with .layers; got {type(inner).__name__}"

    handles = []

    # Per-forward-pass step counter.  Bumped by a forward-pre hook on
    # ``embed_tokens`` (fires exactly once per top-level model invocation).
    # Every per-tensor dump is suffixed with ``.step{i}`` so multi-token
    # generations (max_tokens > 1) produce one dump per decoded step
    # without overwriting earlier steps.
    step = {"i": -1}
    _STEP_STATE["counter"] = step

    def _bump_step(_mod, _args, _kwargs=None):
        step["i"] += 1

    def _save(name):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            # Always copy to CPU/float32 so dumps are diffable across runs.
            dump[f"{name}.step{step['i']}"] = (
                t.detach().to(torch.float32).cpu()
            )
        return _hook

    def _save_rope_call(prefix_template):
        """Forward-PRE hook for a rotary module shared across layers.

        Both vLLM and kb_nano instantiate ``rotary_emb`` once and share it
        across every decoder layer.  A naive forward-pre hook would
        overwrite the dump entry on every call; we use a per-module call
        counter so the Nth call dumps to ``prefix_template.format(N)``.
        """
        def _hook(_mod, args, _kwargs=None):
            try:
                i = getattr(_mod, "_kb_call_count", 0)
                _mod._kb_call_count = i + 1
                positions, query, key = args[0], args[1], args[2]
                p = prefix_template.format(i)
                s = step["i"]
                dump[f"{p}.positions.step{s}"] = positions.detach().to(torch.int64).cpu()
                dump[f"{p}.q_in.step{s}"] = query.detach().to(torch.float32).cpu()
                dump[f"{p}.k_in.step{s}"] = key.detach().to(torch.float32).cpu()
            except Exception:
                pass
        return _hook

    def _save_rope_post(prefix_template):
        """Forward (post) hook for a rotary module shared across layers.

        Pairs with ``_save_rope_call`` to capture the (rotated) outputs of
        the Nth call.  Uses its own counter so the two hooks stay aligned.
        """
        def _hook(_mod, _inp, out):
            try:
                i = getattr(_mod, "_kb_post_call_count", 0)
                _mod._kb_post_call_count = i + 1
                p = prefix_template.format(i)
                s = step["i"]
                if isinstance(out, tuple) and len(out) == 2:
                    q_out, k_out = out
                    dump[f"{p}.q_out.step{s}"] = q_out.detach().to(torch.float32).cpu()
                    dump[f"{p}.k_out.step{s}"] = k_out.detach().to(torch.float32).cpu()
            except Exception:
                pass
        return _hook

    def _save_cache(name, mod):
        """Snapshot the rotary's ``cos_sin_cache`` buffer once."""
        try:
            c = getattr(mod, "cos_sin_cache", None)
            if isinstance(c, torch.Tensor):
                # Slice to first 32 positions to keep the dump small.
                dump[name] = c[:32].detach().to(torch.float32).cpu()
        except Exception:
            pass

    if hasattr(inner, "embed_tokens"):
        # Forward-PRE hook on embed_tokens fires once per model forward —
        # use it to bump the step counter *before* any per-tensor save
        # hook reads ``step['i']``.
        handles.append(inner.embed_tokens.register_forward_pre_hook(
            _bump_step))
        handles.append(inner.embed_tokens.register_forward_hook(
            _save("embed_tokens")))

    layers = inner.layers
    for i in range(min(num_layers, len(layers))):
        layer = layers[i]
        # Whole-layer output (post-MLP, post-residual)
        handles.append(layer.register_forward_hook(_save(f"layer{i}.out")))
        for sub_name in ("input_layernorm", "self_attn",
                         "post_attention_layernorm", "mlp"):
            sub = getattr(layer, sub_name, None)
            if sub is None:
                continue
            handles.append(sub.register_forward_hook(
                _save(f"layer{i}.{sub_name}")))
            # Drill one level deeper into MLA attention so we can isolate
            # which projection diverges first.  Skip ``rotary_emb`` here:
            # it is *shared* across layers (single nn.Module instance), so
            # per-layer hooks would all overwrite each other and produce
            # only the last-call values.  Handled separately below via a
            # call-counter hook.
            if sub_name == "self_attn":
                for proj in ("fused_qkv_a_proj", "q_a_proj",
                             "kv_a_proj_with_mqa", "q_a_layernorm",
                             "q_b_proj", "kv_a_layernorm", "kv_b_proj",
                             "o_proj", "indexer"):
                    p = getattr(sub, proj, None)
                    if p is None:
                        continue
                    handles.append(p.register_forward_hook(
                        _save(f"layer{i}.self_attn.{proj}")))
                # Drill one more level into the indexer (V3.2 only) so we
                # can localise pre-RoPE divergence in wq_b / wk / k_norm.
                indexer = getattr(sub, "indexer", None)
                if indexer is not None:
                    for ip in ("wq_b", "wk", "k_norm", "weights_proj"):
                        pp = getattr(indexer, ip, None)
                        if pp is None:
                            continue
                        handles.append(pp.register_forward_hook(
                            _save(f"layer{i}.self_attn.indexer.{ip}")))
                # Snapshot the absorbed MLA weights so we can compare
                # kb_nano's block-dequant W_UV / W_UK_T against vLLM's
                # ``get_and_maybe_dequant_weights`` (eye-trick) outputs.
                # Search recursively (max depth 4) for any submodule that
                # carries non-empty ``W_UV`` / ``W_UK_T`` attributes — the
                # path differs between engines:
                #   kb_nano: ``self_attn.attn.W_UV``
                #   vLLM:    ``self_attn.mla_attn.mla_attn.W_UV``
                def _snapshot_W(root, depth):
                    if depth < 0 or not isinstance(root, nn.Module):
                        return
                    for w_attr in ("W_UV", "W_UK_T"):
                        w = getattr(root, w_attr, None)
                        if isinstance(w, torch.Tensor) and w.numel():
                            dump[f"layer{i}.self_attn.attn.{w_attr}"] = (
                                w.detach().to(torch.float32).cpu()
                            )
                    for child in root.children():
                        _snapshot_W(child, depth - 1)
                _snapshot_W(sub, depth=4)
            if sub_name == "mlp":
                # MoE submodules of interest. The vLLM ``FusedMoE`` module
                # returns a ``(shared_output, routed_output)`` tuple when
                # the ``shared_experts`` are fused into the kernel; the
                # default ``_save`` hook only captures ``out[0]`` (shared).
                # Use a dedicated tuple-capturing hook for ``experts`` so
                # the routed output is also recorded.
                def _save_moe_tuple(name):
                    def _hook(_mod, _inp, out):
                        try:
                            s = step["i"]
                            if isinstance(out, tuple) and len(out) == 2:
                                if isinstance(out[0], torch.Tensor):
                                    dump[f"{name}.shared_out.step{s}"] = (
                                        out[0].detach().to(torch.float32).cpu()
                                    )
                                if isinstance(out[1], torch.Tensor):
                                    dump[f"{name}.routed_out.step{s}"] = (
                                        out[1].detach().to(torch.float32).cpu()
                                    )
                            elif isinstance(out, torch.Tensor):
                                dump[f"{name}.routed_out.step{s}"] = (
                                    out.detach().to(torch.float32).cpu()
                                )
                        except Exception:
                            pass
                    return _hook
                for moe_sub in ("gate", "shared_experts", "shared_expert",
                                "experts", "grouped_topk"):
                    p = getattr(sub, moe_sub, None)
                    if p is None:
                        continue
                    if moe_sub == "experts":
                        handles.append(p.register_forward_hook(
                            _save_moe_tuple(f"layer{i}.mlp.experts")))
                    else:
                        handles.append(p.register_forward_hook(
                            _save(f"layer{i}.mlp.{moe_sub}")))
                    # Drill into shared expert / dense MLP submodules so we
                    # can localise weight / activation divergence inside the
                    # FP8 MLP pipeline.
                    if moe_sub in ("shared_expert", "shared_experts"):
                        for proj in ("gate_up_proj", "down_proj", "act_fn"):
                            pp = getattr(p, proj, None)
                            if pp is None:
                                continue
                            handles.append(pp.register_forward_hook(
                                _save(f"layer{i}.mlp.{moe_sub}.{proj}")))
                # Dense MLP (first_k_dense_replace = 3 in V3.2): only the
                # outer module exists, so drill into its projections too.
                for proj in ("gate_up_proj", "down_proj", "act_fn"):
                    pp = getattr(sub, proj, None)
                    if pp is None:
                        continue
                    handles.append(pp.register_forward_hook(
                        _save(f"layer{i}.mlp.{proj}")))

    if hasattr(inner, "norm"):
        handles.append(inner.norm.register_forward_hook(_save("final_norm")))

    # ----- Rotary modules.  vLLM shares a single ``rotary_emb`` /
    # ``indexer_rope_emb`` instance across *all* decoder layers (cached by
    # ``get_rope``); kb_nano creates a *new* instance per layer for the
    # indexer.  Per-layer hook registration would either (a) overwrite the
    # dump entry if we keyed by hook-owner layer or (b) under-count if we
    # only tagged the first instance.  Solution: enumerate unique modules
    # in *layer order*, give each one a distinct prefix tag derived from
    # the smallest layer that uses it, and use a per-module call counter.
    main_seen: dict[int, str] = {}
    idx_seen: dict[int, str] = {}
    for i in range(min(num_layers, len(layers))):
        sa = getattr(layers[i], "self_attn", None)
        if sa is None:
            continue
        rope_main = getattr(sa, "rotary_emb", None)
        if rope_main is not None and id(rope_main) not in main_seen:
            tag = f"rope_main_l{i}"
            main_seen[id(rope_main)] = tag
            rope_main._kb_call_count = 0
            rope_main._kb_post_call_count = 0
            handles.append(rope_main.register_forward_pre_hook(
                _save_rope_call(tag + "_call{}")))
            handles.append(rope_main.register_forward_hook(
                _save_rope_post(tag + "_call{}")))
            _save_cache(f"{tag}.cos_sin_cache", rope_main)
        rope_idx = getattr(sa, "indexer_rope_emb", None)
        if rope_idx is not None and id(rope_idx) not in idx_seen:
            tag = f"rope_idx_l{i}"
            idx_seen[id(rope_idx)] = tag
            rope_idx._kb_call_count = 0
            rope_idx._kb_post_call_count = 0
            handles.append(rope_idx.register_forward_pre_hook(
                _save_rope_call(tag + "_call{}")))
            handles.append(rope_idx.register_forward_hook(
                _save_rope_post(tag + "_call{}")))
            _save_cache(f"{tag}.cos_sin_cache", rope_idx)

    # Monkey-patch MoE forward to capture internal tensors.  We restore
    # the original method via a handle-like object the caller can ``remove``.
    class _Restorer:
        def __init__(self, obj, attr, orig):
            self._obj, self._attr, self._orig = obj, attr, orig

        def remove(self):
            setattr(self._obj, self._attr, self._orig)

    # ----- Engine-specific debug instrumentation -------------------------
    # The sparse_fwd input/output hook + MLA dispatch traces are only
    # useful when running kb_nano in-process. For vLLM we run inside an
    # EngineCore worker via cloudpickle. Even ``import`` statements inside
    # ``install_layer_hooks`` add their dotted-name tokens (e.g.
    # ``backends``, ``third_party``) to ``co_names``, which causes
    # cloudpickle's ``_find_imported_submodules`` to drag in
    # ``torch.backends`` (a non-picklable ``GenericModule``). Therefore
    # all engine-specific patching is delegated to helpers defined at
    # module level so their imports do not pollute this function's
    # ``co_names``.
    # Install the sparse_fwd input/output hook for both engines so we
    # can directly compare what flows into FlashMLA's sparse kernel.
    # The helper uses ``importlib`` so cloudpickle doesn't chase the
    # dotted-name submodules.
    _install_kb_nano_sparse_fwd_hook(dump, handles)
    _install_dsv3_router_gemm_hook(dump, handles)
    _install_grouped_topk_hook(dump, handles)
    _install_deepgemm_grouped_hook(dump, handles)
    if engine != "kb_nano":
        _install_vllm_mla_dispatch_traces()
        return handles

    _install_kb_nano_mla_dispatch_traces()
    _install_vllm_mla_dispatch_traces()

    def _save_tensor(name, t):
        if isinstance(t, torch.Tensor):
            s = _step()
            dump[f"{name}.step{s}"] = t.detach().to(torch.float32).cpu()

    for i in range(min(num_layers, len(layers))):
        layer = layers[i]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        kind = type(mlp).__name__
        # ----- kb_nano: DeepSeekMoE.forward_impl --------------------------
        if kind == "DeepSeekMoE" and hasattr(mlp, "forward_impl"):
            orig_impl = mlp.forward_impl
            tag = f"layer{i}.mlp_internal"

            def _make_kb_wrap(mlp_ref, orig, prefix):
                _printed = {"once": False}

                def _wrapped(hidden_states):
                    orig_shape = hidden_states.shape
                    hs = hidden_states.view(-1, mlp_ref.hidden_size)
                    if not _printed["once"]:
                        _printed["once"] = True
                        print(f"[diag] {prefix}: grouped_topk type = "
                              f"{type(mlp_ref.grouped_topk).__name__}, "
                              f"scoring_func = "
                              f"{getattr(mlp_ref.grouped_topk, 'scoring_func', '?')}, "
                              f"renormalize = "
                              f"{getattr(mlp_ref.grouped_topk, 'renormalize', '?')}, "
                              f"_use_custom_op = {mlp_ref._use_custom_op}",
                              flush=True)
                    _save_tensor(f"{prefix}.hidden_in", hs)
                    # Replicate the original logic with capture points:
                    use_shared_stream = (
                        mlp_ref.shared_expert is not None
                        and not mlp_ref._disable_shared_stream
                    )
                    shared_out = None
                    if use_shared_stream:
                        if mlp_ref._shared_stream is None:
                            mlp_ref._shared_stream = torch.cuda.Stream()
                        mlp_ref._shared_stream.wait_stream(
                            torch.cuda.current_stream())
                        with torch.cuda.stream(mlp_ref._shared_stream):
                            shared_out = mlp_ref.shared_expert(hs)
                        torch.cuda.current_stream().wait_stream(
                            mlp_ref._shared_stream)
                    elif mlp_ref.shared_expert is not None:
                        shared_out = mlp_ref.shared_expert(hs)
                    if shared_out is not None:
                        _save_tensor(f"{prefix}.shared_out", shared_out)

                    # Mirror DeepSeekMoE.forward_impl exactly so the diagnostic
                    # exercises the same gate matmul kernels as the real model
                    # (DSV3 specialized / cuBLAS BF16->FP32). Promoting both
                    # sides to FP32 before matmul produces a slightly different
                    # accumulation order which flips near-tie expert / group
                    # selections in the noaux_tc grouped-topk path.
                    from kb_nano.tasks.baseline.L1.gate_linear import (
                        gate_linear_forward,
                    )
                    # Match DeepSeekMoE.forward_impl: BF16 out when FP8
                    # experts (vLLM's non-monolithic path), FP32 otherwise.
                    router_out_dtype = (
                        torch.float32 if not mlp_ref.use_fp8
                        else torch.bfloat16
                    )
                    router_logits = gate_linear_forward(
                        hs, mlp_ref.gate_weight, out_dtype=router_out_dtype,
                    )
                    _save_tensor(f"{prefix}.router_logits", router_logits)
                    topk_weights, topk_ids = mlp_ref.grouped_topk(
                        router_logits, mlp_ref.e_score_correction_bias,
                        mlp_ref.n_group, mlp_ref.topk_group, mlp_ref.top_k,
                    )
                    _save_tensor(f"{prefix}.topk_weights", topk_weights)
                    _save_tensor(f"{prefix}.topk_ids", topk_ids)

                    # Match DeepSeekMoE.forward_impl: FP32 topk_weights to
                    # the FP8 path (vLLM's invoke_fused_moe_triton_kernel
                    # consumes FP32 weights), BF16 to the unquantised path.
                    if mlp_ref.use_fp8:
                        routed = mlp_ref._forward_fp8_experts(
                            hs, topk_weights, topk_ids)
                    else:
                        topk_weights_act = topk_weights.to(hs.dtype)
                        routed = mlp_ref.fused_experts(
                            hs, mlp_ref.w13, mlp_ref.w2,
                            topk_weights_act, topk_ids, mlp_ref.num_experts,
                        )
                    _save_tensor(f"{prefix}.routed_out_preScale", routed)
                    routed = routed * mlp_ref.routed_scaling_factor
                    _save_tensor(f"{prefix}.routed_out_postScale", routed)

                    out = routed
                    if shared_out is not None:
                        out = out + shared_out
                    _save_tensor(f"{prefix}.preAllreduce", out)

                    if mlp_ref.tp_size > 1:
                        out = mlp_ref.allreduce(out)

                    return out.view(orig_shape)
                return _wrapped

            mlp.forward_impl = _make_kb_wrap(mlp, orig_impl, tag)
            handles.append(_Restorer(mlp, "forward_impl", orig_impl))

        # ----- vLLM: DeepseekV2MoE.forward --------------------------------
        elif kind == "DeepseekV2MoE":
            orig_fwd = mlp.forward
            tag = f"layer{i}.mlp_internal"

            def _make_vllm_wrap(mlp_ref, orig, prefix):
                def _wrapped(hidden_states):
                    num_tokens, hidden_dim = hidden_states.shape
                    hs = hidden_states.view(-1, hidden_dim)
                    _save_tensor(f"{prefix}.hidden_in", hs)

                    if mlp_ref.experts.is_internal_router:
                        fused_moe_out = mlp_ref.experts(
                            hidden_states=hs, router_logits=hs)
                    else:
                        router_logits, _ = mlp_ref.gate(hs)
                        _save_tensor(f"{prefix}.router_logits", router_logits)
                        fused_moe_out = mlp_ref.experts(
                            hidden_states=hs, router_logits=router_logits)
                    shared_output, final_hidden_states = fused_moe_out
                    _save_tensor(f"{prefix}.routed_out_preScale",
                                 final_hidden_states)
                    if shared_output is not None:
                        _save_tensor(f"{prefix}.shared_out", shared_output)

                    if hidden_states.dtype != torch.float16:
                        if not mlp_ref.is_rocm_aiter_moe_enabled:
                            final_hidden_states = (
                                final_hidden_states *
                                mlp_ref.routed_scaling_factor
                            )
                    _save_tensor(f"{prefix}.routed_out_postScale",
                                 final_hidden_states)
                    if mlp_ref.shared_experts is not None:
                        final_hidden_states = (
                            final_hidden_states + shared_output
                        )
                    _save_tensor(f"{prefix}.preAllreduce",
                                 final_hidden_states)

                    if mlp_ref.tp_size > 1:
                        final_hidden_states = (
                            mlp_ref.experts
                            .maybe_all_reduce_tensor_model_parallel(
                                final_hidden_states)
                        )
                    return final_hidden_states.view(num_tokens, hidden_dim)
                return _wrapped

            mlp.forward = _make_vllm_wrap(mlp, orig_fwd, tag)
            handles.append(_Restorer(mlp, "forward", orig_fwd))

    return handles


# ---------------------------------------------------------------------------
# kb_nano runner
# ---------------------------------------------------------------------------
def run_kb_nano(model_path: str, output_path: str, num_layers: int,
                tp: int = 1, max_tokens: int = 1) -> None:
    print(f"[kb_nano] importing engine...", flush=True)
    # Monkey-patch download_model so kb_nano accepts our local staged path.
    from kb_nano.infra import weight_loader as _wl
    _wl.download_model = lambda name: name if os.path.isdir(name) else _wl.snapshot_download(
        name, allow_patterns=["*.safetensors", "*.json"])
    # GDS doesn't work on symlinked staging dirs.  At TP=1 we always go
    # through the GDS path of fastsafetensors, so disable the library
    # outright.  At TP>1, fastsafetensors automatically falls back to a
    # CPU-staged broadcast (``nogds=True``) which is symlink-safe and we
    # MUST keep it enabled — otherwise rank 0 (patched here) would skip
    # the collective broadcasts that the spawned worker ranks (which do
    # not inherit this patch) are still issuing, causing a deadlock at
    # weight-load time.
    if tp == 1:
        _wl._HAS_FASTSAFETENSORS = False
    # Patch huggingface_hub.hf_hub_download so the fallback DeepSeek-V3.2 config
    # loader (which raises ValueError on transformers and then tries
    # hf_hub_download) also accepts our local path.
    import huggingface_hub
    _orig_hub_download = huggingface_hub.hf_hub_download
    def _patched_hub(repo_id, filename, **kwargs):
        if os.path.isdir(repo_id):
            local = os.path.join(repo_id, filename)
            if os.path.exists(local):
                return local
        return _orig_hub_download(repo_id, filename, **kwargs)
    huggingface_hub.hf_hub_download = _patched_hub
    # Same patch in the loader module's already-imported reference.
    if hasattr(_wl, "hf_hub_download"):
        _wl.hf_hub_download = _patched_hub

    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    engine = LlamaEngine(
        model_name=model_path,
        seed=SEED,
        enforce_eager=True,    # disable cuda graph so hooks fire
        tensor_parallel_size=tp,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        # Keep warmup fast — the diagnostic only ever processes one
        # short prompt, and a heavy warmup (8192 tokens × 256 seqs)
        # triggers long Triton autotuning per unique shape, which at
        # TP=8 has exceeded 15 minutes in testing.
        max_num_batched_tokens=512,
        max_num_seqs=4,
    )

    # Reach into the model runner to install hooks on the underlying model.
    runner = engine.model_runner
    model = runner.model
    dump: dict = {}
    handles = install_layer_hooks(model, dump, num_layers,
                                  prefix_path=["model"], engine="kb_nano")
    print(f"[kb_nano] installed {len(handles)} forward hooks", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                        ignore_eos=True)
    out = engine.generate(
        prompts=[PROMPT_TOKEN_IDS],
        sampling_params=[sp],
        use_tqdm=False,
    )
    gen_ids = list(out[0].token_ids) if out[0].token_ids else []
    print(f"[kb_nano] forward complete; generated ids = {gen_ids}",
          flush=True)

    for h in handles:
        h.remove()

    print(f"[kb_nano] dumping {len(dump)} tensors to {output_path}",
          flush=True)
    _save_dump_parallel(dump, output_path, engine="kb_nano")


# ---------------------------------------------------------------------------
# vLLM runner
# ---------------------------------------------------------------------------
def _vllm_install_hooks_worker(model):
    """Runs *inside* the vLLM EngineCore worker process (TP=1).

    We can't ship a closure over a local dict back to the parent because
    ``apply_model`` round-trips through msgspec serialisation.  Instead, we
    stash the dump dict on the model itself (as a regular Python attribute)
    and rely on a second ``apply_model`` call after generation to dump it
    to disk.
    """
    import os
    num_layers = int(os.environ["KB_DIFF_NUM_LAYERS"])
    dump: dict = {}
    # Attach to the model so a later apply_model call can fetch it.
    model._kb_dump = dump
    handles = install_layer_hooks(model, dump, num_layers,
                                  prefix_path=["model"], engine="vllm")

    # Monkey-patch the grouped-topk router so we can compare topk_ids /
    # topk_weights with kb_nano's reference grouped_topk.
    import vllm.model_executor.layers.fused_moe.router.grouped_topk_router as _gtr  # type: ignore
    _orig_fgt = _gtr.fused_grouped_topk
    _call_idx = {"i": 0}

    # Number of MoE layers per forward = total layers - first_k_dense_replace.
    # In V3.2 there are 58 MoE layers (61 - 3).  We deduce ``num_moe_layers``
    # the first time we see it become known via the model config; until then
    # we fall back to the total layer count from KB_DIFF_NUM_LAYERS - 3.
    _num_moe = max(1, int(os.environ.get("KB_DIFF_NUM_LAYERS", "61")) - 3)

    def _wrapped_fgt(*args, **kwargs):
        out = _orig_fgt(*args, **kwargs)
        idx = _call_idx["i"]
        _call_idx["i"] += 1
        # The wrapped fn is called once per MoE layer per forward step, in
        # layer order.  Map global call idx -> (step, moe_idx) by dividing
        # by the number of MoE layers; then convert moe_idx -> layer id by
        # adding ``first_k_dense_replace`` (3 for V3.2).
        s = idx // _num_moe
        moe_idx = idx % _num_moe
        layer_id = moe_idx + 3
        try:
            tw, ti = out
            dump[f"layer{layer_id}.mlp_internal.topk_weights.step{s}"] = (
                tw.detach().to(torch.float32).cpu()
            )
            dump[f"layer{layer_id}.mlp_internal.topk_ids.step{s}"] = (
                ti.detach().to(torch.int32).cpu()
            )
            # Capture the gate logits going into the router so we can
            # see how close kb_nano and vLLM are before the top-k pick.
            # fused_grouped_topk signature:
            #   (hidden_states, gating_output, topk, renormalize,
            #    e_score_correction_bias, ...)
            gating = kwargs.get("gating_output")
            if gating is None and len(args) >= 2 and isinstance(args[1], torch.Tensor):
                gating = args[1]
            if isinstance(gating, torch.Tensor):
                dump[f"layer{layer_id}.mlp_internal.router_logits.step{s}"] = (
                    gating.detach().to(torch.float32).cpu()
                )
        except Exception:
            pass
        return out

    _gtr.fused_grouped_topk = _wrapped_fgt

    class _PatchRestorer:
        def remove(self):
            _gtr.fused_grouped_topk = _orig_fgt

    handles.append(_PatchRestorer())

    model._kb_handles = handles
    return len(handles)


def _vllm_save_dump_worker(model):
    import os
    # Only rank 0 writes the output file. The tensors we compare (router
    # logits, topk_ids, topk_weights, full-layer outputs after TP
    # all-reduce, DSA indices) are replicated across TP ranks, so rank 0
    # alone is sufficient.
    try:
        import torch.distributed as _td
        rank = _td.get_rank() if _td.is_initialized() else 0
    except Exception:
        rank = 0
    output_path = os.environ["KB_DIFF_OUTPUT_PATH"]
    dump = getattr(model, "_kb_dump", {})
    if rank == 0:
        _save_dump_parallel(dump, output_path, engine="vllm")
    for h in getattr(model, "_kb_handles", []):
        h.remove()
    return len(dump) if rank == 0 else 0


def run_vllm(model_path: str, output_path: str, num_layers: int,
             tp: int = 1, max_tokens: int = 1) -> None:
    print(f"[vllm] importing engine...", flush=True)
    # Workers will read these out of the env so we don't need to send the
    # values across the EngineCore boundary.
    os.environ["KB_DIFF_NUM_LAYERS"] = str(num_layers)
    os.environ["KB_DIFF_OUTPUT_PATH"] = output_path
    # The EngineCore subprocess (used at TP>1) re-imports our worker
    # callbacks via ``kb_nano.tests.debug._vllm_diff_workers``; that
    # trampoline needs to know where this script lives on disk so it can
    # load it back as a regular module (it cannot pickle ``__main__``).
    os.environ["KB_DIFF_SCRIPT_PATH"] = str(Path(__file__).resolve())

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        enforce_eager=True,
        seed=SEED,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        load_format="fastsafetensors",
    )

    # Use the importable trampolines so vLLM's stdlib-pickle shm broadcast
    # at TP>1 can resolve the callback in the EngineCore worker.  At TP=1
    # this is also fine because the trampoline simply forwards back to
    # ``_vllm_install_hooks_worker`` defined in this module.
    from kb_nano.tests.debug import _vllm_diff_workers as _wk
    n_hooks = llm.apply_model(_wk.install_hooks)
    print(f"[vllm] installed {n_hooks} forward hooks (per rank)", flush=True)

    sp = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens, ignore_eos=True,
    )
    from vllm.inputs import TokensPrompt
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=PROMPT_TOKEN_IDS)],
        sampling_params=sp, use_tqdm=False,
    )
    gen_ids = list(outs[0].outputs[0].token_ids) if outs[0].outputs[0].token_ids else []
    print(f"[vllm] forward complete; generated ids = {gen_ids}",
          flush=True)

    n_saved = llm.apply_model(_wk.save_dump)
    print(f"[vllm] dumped {n_saved} tensors to {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------
def diff_dumps(path_a: str, path_b: str) -> None:
    _, ta = _load_dump_parallel(path_a)
    _, tb = _load_dump_parallel(path_b)
    common = sorted(set(ta) & set(tb))
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))

    print(f"\n{'name':<60s}  {'shape':<22s} {'cos':>10s} {'mse':>11s} {'max|diff|':>12s}")
    print("-" * 120)

    def _stats(x: torch.Tensor, y: torch.Tensor):
        x = x.float().flatten()
        y = y.float().flatten()
        if x.numel() != y.numel():
            return None
        cos = float(torch.nn.functional.cosine_similarity(
            x.unsqueeze(0), y.unsqueeze(0)))
        mse = float(((x - y) ** 2).mean())
        mx = float((x - y).abs().max())
        return cos, mse, mx

    for n in common:
        s = _stats(ta[n], tb[n])
        if s is None:
            print(f"{n:<60s}  shape mismatch a={tuple(ta[n].shape)} "
                  f"b={tuple(tb[n].shape)}")
            continue
        cos, mse, mx = s
        marker = "" if cos > 0.999 else (" <<<" if cos < 0.99 else " <")
        print(f"{n:<60s}  {str(tuple(ta[n].shape)):<22s} "
              f"{cos:>10.6f} {mse:>11.2e} {mx:>12.4f}{marker}")

    if only_a:
        print(f"\nOnly in {path_a}:")
        for n in only_a:
            print(f"  {n}")
    if only_b:
        print(f"\nOnly in {path_b}:")
        for n in only_b:
            print(f"  {n}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["kb_nano", "vllm"])
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--tp", type=int, default=1,
                   help="tensor-parallel size (1 or 8)")
    p.add_argument("--max-tokens", type=int, default=1,
                   help="number of tokens to generate (>=1).  With N>1 the "
                        "diagnostic captures each step's activations under "
                        "a `.step{i}` suffix so prefill / decode-step-k "
                        "divergence can be compared independently.")
    p.add_argument("--output", type=str)
    p.add_argument("--diff", nargs=2, metavar=("A", "B"))
    args = p.parse_args()

    if args.diff is not None:
        diff_dumps(args.diff[0], args.diff[1])
        return

    if args.engine is None or args.output is None:
        p.error("--engine and --output are required when not using --diff")

    model_path = make_truncated_checkpoint_dir(args.model, args.num_layers)

    if args.engine == "kb_nano":
        run_kb_nano(model_path, args.output, args.num_layers, args.tp,
                    max_tokens=args.max_tokens)
    else:
        run_vllm(model_path, args.output, args.num_layers, args.tp,
                 max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
