"""Isolated kernel-level benchmarking via direct forward() calls.

Instantiates baseline and candidate nn.Module instances, copies weights,
loads inputs from the InputRegistry (random or golden), compares outputs
and timing. No full model build required — per-kernel test time is seconds
rather than minutes.
"""

from __future__ import annotations

import time
import inspect
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from kb_nano.bench.kernels.scenario_registry import InputRegistry
from kb_nano.infra.kernel_swapper import (
    BenchTarget,
    discover_references,
    discover_targets,
    get,
    load_candidate,
    load_reference,
)

from .result import KernelBenchResult, OperatorResult, ScenarioResult

_DEFAULT_REGISTRY = None
_FP32_ATOL = 1e-5
_FP32_RTOL = 1e-3
_LOW_PRECISION_ATOL = 1e-2
_LOW_PRECISION_RTOL = 1e-2
_FP8_ATOL = 1.25e-1
_FP8_RTOL = 1.25e-1
_FP8_GROUP_SIZE = 128
_FP8_WEIGHT_CACHE: dict[tuple[str, int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_MXFP4_BLOCK_SIZE = 32
_MXFP4_RAW_WEIGHT_CACHE: dict[tuple[int, int, int, torch.device, str], dict[str, torch.Tensor]] = {}
_MXFP4_PREPARED_WEIGHT_CACHE: dict[
    tuple[type, int, int, int, torch.device, str],
    tuple[Any, Any, Any],
] = {}


def _short_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"


def _get_registry() -> InputRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = InputRegistry()
    return _DEFAULT_REGISTRY


def _find_candidate_path(target_name: str, level: int) -> str:
    """Return the relative path to the candidate file for display."""
    return f"tasks/candidate/L{level}/{target_name}.py"


def _find_reference_path(target_name: str, level: int) -> str:
    """Return the relative path to the semantic reference file for display."""
    return f"tasks/reference/L{level}/{target_name}.py"


def _instantiate_module(
    cls: type,
    init_args: dict[str, Any],
    inputs: dict[str, Any] | None = None,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Create an nn.Module instance with init_args, handling common patterns."""
    kwargs = dict(init_args)
    class_name = cls.__name__
    if "head_size" in kwargs and "head_dim" not in kwargs and class_name != "Attention":
        kwargs["head_dim"] = kwargs.pop("head_size")
    if "head_dim" in kwargs and class_name == "FluxAttention" and "dim_head" not in kwargs:
        kwargs["dim_head"] = kwargs.pop("head_dim")
    if "base" in kwargs and "rope_theta" not in kwargs:
        kwargs["rope_theta"] = kwargs.pop("base")
    if "variance_epsilon" in kwargs and "eps" not in kwargs:
        kwargs["eps"] = kwargs.pop("variance_epsilon")
    if "rotary_dim" in kwargs and "head_dim" not in kwargs:
        kwargs["head_dim"] = kwargs["rotary_dim"]
    if "org_vocab_size" in kwargs and "org_num_embeddings" not in kwargs:
        kwargs["org_num_embeddings"] = kwargs.pop("org_vocab_size")
    if "use_bias" in kwargs and "bias" not in kwargs:
        kwargs["bias"] = kwargs.pop("use_bias")
    if "d" in kwargs and "dimension" not in kwargs:
        kwargs["dimension"] = kwargs.pop("d")
    if kwargs.get("use_fp8") and "quant_config" not in kwargs:
        kwargs["quant_config"] = {"weight_block_size": kwargs.get("block_shape", [128, 128])}
    kwargs.pop("rotary_dim", None)
    kwargs.pop("is_neox_style", None)
    kwargs.pop("training", None)
    kwargs.pop("tp_rank", None)
    kwargs.pop("tp_size", None)
    kwargs.pop("use_fp8", None)

    if inputs is not None:
        if class_name == "Linear" and "input" in inputs:
            kwargs.setdefault("in_features", int(inputs["input"].shape[-1]))
            kwargs.setdefault("out_features", int(inputs["input"].shape[-1]))
        elif class_name == "Embedding" and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            max_id = int(input_ids.max().item()) if input_ids.numel() else 0
            kwargs.setdefault("num_embeddings", max(128, max_id + 1))
            kwargs.setdefault("embedding_dim", 128)
        elif class_name == "Conv2d" and "x" in inputs:
            x = inputs["x"]
            in_channels = int(x.shape[1])
            groups = int(kwargs.get("groups", 1))
            kwargs.setdefault("in_channels", in_channels)
            kwargs.setdefault("out_channels", in_channels)
            kwargs.setdefault("kernel_size", 1)
            kwargs["groups"] = max(1, min(groups, in_channels))
        elif class_name == "LayerNorm":
            normalized_shape = kwargs.get("normalized_shape")
            if isinstance(normalized_shape, list) and len(normalized_shape) == 1:
                kwargs["normalized_shape"] = int(normalized_shape[0])
        elif class_name == "T5LayerNorm" and "hidden_states" in inputs:
            kwargs.setdefault("hidden_size", int(inputs["hidden_states"].shape[-1]))
        elif class_name == "VisionRotaryEmbedding":
            kwargs.setdefault("rotary_dim", 40)
        elif class_name in ("RotaryEmbedding", "MRotaryEmbedding"):
            positions = inputs.get("positions")
            max_position = 16
            if isinstance(positions, torch.Tensor) and positions.numel() > 0:
                max_position = max(max_position, int(positions.max().item()) + 1)
            query = inputs.get("query")
            if "head_dim" not in kwargs and isinstance(query, torch.Tensor):
                kwargs["head_dim"] = int(query.shape[-1])
            kwargs.setdefault("max_position_embeddings", max_position)
            kwargs.setdefault("rope_theta", 10000.0)
            if class_name == "MRotaryEmbedding":
                head_dim = int(kwargs.get("head_dim", 128))
                half = head_dim // 2
                base = half // 3
                kwargs.setdefault("mrope_section", [half - 2 * base, base, base])
        elif class_name == "AdaLayerNormZeroSingle" and "x" in inputs:
            kwargs.setdefault("embedding_dim", int(inputs["x"].shape[-1]))
        elif class_name == "AdaLayerNormContinuous" and "x" in inputs:
            kwargs.setdefault("embedding_dim", int(inputs["x"].shape[-1]))
            cond = inputs.get("conditioning_embedding")
            if isinstance(cond, torch.Tensor):
                kwargs.setdefault("conditioning_embedding_dim", int(cond.shape[-1]))
        elif class_name in ("BertEmbeddings", "XLMRobertaEmbeddings"):
            hidden = 1024
            embeds = inputs.get("inputs_embeds")
            if isinstance(embeds, torch.Tensor):
                hidden = int(embeds.shape[-1])
            kwargs["config"] = SimpleNamespace(
                vocab_size=128,
                hidden_size=hidden,
                max_position_embeddings=128,
                type_vocab_size=2,
                layer_norm_eps=1e-5,
                pad_token_id=int(kwargs.get("padding_idx", 1)),
                position_embedding_type=kwargs.get("position_embedding_type", "absolute"),
            )
        elif class_name == "EncoderOutput" and "hidden_states" in inputs and "input_tensor" in inputs:
            kwargs["config"] = SimpleNamespace(
                hidden_size=int(inputs["input_tensor"].shape[-1]),
                intermediate_size=int(inputs["hidden_states"].shape[-1]),
                layer_norm_eps=1e-5,
            )
        elif class_name == "FeedForward" and "hidden_states" in inputs:
            kwargs.setdefault("dim", int(inputs["hidden_states"].shape[-1]))
        elif class_name == "GLAMLP" and "x" in inputs:
            hidden = int(inputs["x"].shape[-1])
            kwargs.setdefault("hidden_size", hidden)
            kwargs.setdefault("intermediate_size", hidden * 4)
        elif class_name == "GatedLinearAttention" and "hidden_states" in inputs:
            hidden = int(inputs["hidden_states"].shape[-1])
            kwargs.setdefault("hidden_size", hidden)
            kwargs.setdefault("num_heads", int(kwargs.get("num_heads", 1)))
            if "key_dim" in kwargs:
                kwargs.setdefault("expand_k", float(kwargs["key_dim"]) / max(1, hidden))
            if "value_dim" in kwargs:
                kwargs.setdefault("expand_v", float(kwargs["value_dim"]) / max(1, hidden))
            kwargs.pop("key_dim", None)
            kwargs.pop("value_dim", None)
            kwargs.pop("head_k_dim", None)
            kwargs.pop("head_v_dim", None)
        elif class_name == "LlamaMLP" and "x" in inputs:
            hidden = int(inputs["x"].shape[-1])
            intermediate = 14336 if hidden == 4096 else hidden * 4
            kwargs["config"] = SimpleNamespace(hidden_size=hidden, intermediate_size=intermediate)
        elif class_name == "GptOssMoE" and "hidden_states" in inputs:
            hidden = int(kwargs.get("hidden_size", inputs["hidden_states"].shape[-1]))
            kwargs["config"] = SimpleNamespace(
                num_local_experts=int(kwargs.get("num_experts", 128)),
                num_experts_per_tok=int(kwargs.get("top_k", 4)),
                hidden_size=hidden,
                intermediate_size=int(kwargs.get("intermediate_per_tp", hidden * 2)),
            )
        elif class_name == "Qwen3MoE" and "hidden_states" in inputs:
            hidden = int(kwargs.get("hidden_size", inputs["hidden_states"].shape[-1]))
            kwargs["config"] = SimpleNamespace(
                num_experts=int(kwargs.get("num_experts", 128)),
                num_experts_per_tok=int(kwargs.get("top_k", 8)),
                hidden_size=hidden,
                moe_intermediate_size=int(kwargs.get("intermediate_per_tp", hidden // 4)),
                norm_topk_prob=bool(kwargs.get("renormalize", True)),
            )
        elif class_name == "OasisFinalLayer" and "x" in inputs:
            kwargs.setdefault("hidden_size", int(inputs["x"].shape[-1]))
            kwargs.setdefault("patch_size", 2)
            kwargs.setdefault("out_channels", 16)
        elif class_name == "OasisMLP" and "x" in inputs:
            hidden = int(inputs["x"].shape[-1])
            kwargs.setdefault("in_features", hidden)
            kwargs.setdefault("hidden_features", hidden * 4)
            kwargs.setdefault("out_features", hidden)
        elif class_name == "OasisTimestepEmbedder":
            kwargs.setdefault("hidden_size", 1024)
        elif class_name == "OasisPatchEmbed" and "x" in inputs:
            x = inputs["x"]
            patch = kwargs.get("patch_size", 2)
            if isinstance(patch, list):
                patch = int(patch[0])
            kwargs.setdefault("img_height", int(x.shape[-2]))
            kwargs.setdefault("img_width", int(x.shape[-1]))
            kwargs["patch_size"] = int(patch)
            kwargs.setdefault("in_chans", int(x.shape[1]))
            kwargs.setdefault("embed_dim", 1024)
        elif class_name in ("OasisSpatialAxialAttention", "OasisTemporalAxialAttention") and "x" in inputs:
            hidden = int(inputs["x"].shape[-1])
            heads = int(kwargs.get("heads", 16))
            dim_head = hidden // heads
            kwargs.setdefault("dim", hidden)
            kwargs.setdefault("dim_head", dim_head)
            if "rotary_emb" not in kwargs:
                from kb_nano.tasks.baseline.L1.oasis_rotary import OasisRotaryEmbedding

                rotary_dim = dim_head // 4 if class_name == "OasisSpatialAxialAttention" else dim_head
                freqs_for = "pixel" if class_name == "OasisSpatialAxialAttention" else "lang"
                kwargs["rotary_emb"] = OasisRotaryEmbedding(dim=rotary_dim, freqs_for=freqs_for)
        elif class_name == "OasisVAEAttention" and "x" in inputs:
            kwargs.setdefault("dim", int(inputs["x"].shape[-1]))
        elif class_name == "CombinedTimestepGuidanceTextProjEmbeddings" and "pooled_projection" in inputs:
            kwargs.setdefault("embedding_dim", 3072)
            kwargs.setdefault("pooled_projection_dim", int(inputs["pooled_projection"].shape[-1]))
        elif class_name == "VisionAttention" and "x" in inputs:
            x = inputs["x"]
            kwargs.setdefault("embed_dim", int(x.shape[-1]))
            if "head_dim" in kwargs and "num_heads" in kwargs:
                kwargs.setdefault("projection_size", int(kwargs["head_dim"]) * int(kwargs["num_heads"]))
            kwargs.pop("head_dim", None)
        elif class_name == "VisionMLP" and "x" in inputs:
            hidden = int(inputs["x"].shape[-1])
            kwargs.setdefault("in_features", hidden)
            kwargs.setdefault("hidden_features", 4304 if hidden == 1152 else hidden * 4)
        elif class_name == "VisionPatchEmbed" and "x" in inputs:
            input_size = int(kwargs.get("input_size", inputs["x"].shape[-1]))
            kwargs.setdefault("patch_size", 1)
            kwargs.setdefault("temporal_patch_size", 1)
            kwargs.setdefault("in_channels", input_size)
            kwargs.setdefault("embed_dim", int(kwargs.get("embed_dim", 1152)))
        elif class_name == "VisionPatchMerger" and "x" in inputs:
            context_dim = int(inputs["x"].shape[-1])
            hidden_size = int(kwargs.get("hidden_size", context_dim * 4))
            spatial_merge = max(1, int(round((hidden_size / context_dim) ** 0.5)))
            kwargs.setdefault("d_model", context_dim)
            kwargs.setdefault("context_dim", context_dim)
            kwargs.setdefault("spatial_merge_size", spatial_merge)
        elif class_name == "VisionPosEmbedInterpolate":
            grid = int(kwargs.get("num_grid_per_side", 48))
            kwargs.setdefault("num_position_embeddings", grid * grid)
        elif class_name == "YOLOAttention" and "x" in inputs:
            c = int(inputs["x"].shape[1])
            kwargs.setdefault("dim", c)
            if "head_dim" in kwargs and "key_dim" in kwargs:
                kwargs.setdefault("attn_ratio", float(kwargs["key_dim"]) / max(1, float(kwargs["head_dim"])))
            kwargs.pop("head_dim", None)
            kwargs.pop("key_dim", None)
            kwargs.pop("scale", None)
        elif class_name in ("YOLOBottleneck", "YOLOCIB", "YOLOSPPF") and "x" in inputs:
            c = int(inputs["x"].shape[1])
            kwargs.setdefault("c1", c)
            kwargs.setdefault("c2", c)
            kwargs.pop("add", None)
        elif class_name == "YOLOC2fCIB" and "x" in inputs:
            c_mid = int(kwargs.get("c", max(1, inputs["x"].shape[1] // 2)))
            kwargs.setdefault("c1", int(inputs["x"].shape[1]))
            kwargs.setdefault("c2", c_mid * 2)
            kwargs.pop("c", None)
        elif class_name == "YOLOConv" and "x" in inputs:
            c = int(inputs["x"].shape[1])
            kwargs.setdefault("c1", c)
            kwargs.setdefault("c2", c)
        elif class_name == "YOLOPSA" and "x" in inputs:
            c = int(inputs["x"].shape[1])
            kwargs.setdefault("c1", c)
            kwargs.setdefault("c2", c)
            if "c" in kwargs:
                kwargs.setdefault("e", float(kwargs.pop("c")) / max(1, c))
        elif class_name == "YOLORepVGGDW" and "x" in inputs:
            kwargs.setdefault("ed", int(inputs["x"].shape[1]))
        elif class_name == "YOLOSCDown" and "x" in inputs:
            c = int(inputs["x"].shape[1])
            kwargs.setdefault("c1", c)
            kwargs.setdefault("c2", c * 2)
            kwargs.setdefault("k", 3)
            kwargs.setdefault("s", 2)
        elif class_name == "RowParallelLinear" and "x" in inputs:
            input_size = int(inputs["x"].shape[-1])
            kwargs.setdefault("input_size", input_size)
            if input_size == 12288:
                output_size = 3072
            elif input_size == 14336:
                output_size = 4096
            else:
                output_size = input_size
            kwargs.setdefault("output_size", output_size)
        elif class_name == "LlamaAttention" and "hidden_states" in inputs:
            hidden = int(inputs["hidden_states"].shape[-1])
            kwargs.setdefault("hidden_size", hidden)
            if "num_heads" in kwargs and "num_attention_heads" not in kwargs:
                kwargs["num_attention_heads"] = kwargs.pop("num_heads")
            if "num_kv_heads" in kwargs and "num_key_value_heads" not in kwargs:
                kwargs["num_key_value_heads"] = kwargs.pop("num_kv_heads")
            kwargs.setdefault("qk_norm", kwargs.get("q_norm") is not None and kwargs.get("k_norm") is not None)
            kwargs.setdefault("use_weightless_qk_norm", kwargs.get("q_wl_norm") is not None and kwargs.get("k_wl_norm") is not None)
            kwargs.setdefault("use_sinks", kwargs.get("sinks") is not None)
            kwargs.pop("q_norm", None)
            kwargs.pop("k_norm", None)
            kwargs.pop("q_wl_norm", None)
            kwargs.pop("k_wl_norm", None)
            kwargs.pop("sinks", None)
        elif class_name == "FluxAttention":
            kwargs.pop("inner_dim", None)

    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )
        if not accepts_kwargs:
            kwargs = {
                k: v for k, v in kwargs.items()
                if k in params and k != "self"
            }
    except (TypeError, ValueError):
        pass

    try:
        module = cls(**kwargs)
    except TypeError:
        module = cls()

    module = module.to(device)
    if dtype is not None:
        # Cast learnable parameters to the scenario dtype without changing
        # precision-sensitive buffers such as RoPE/YARN cos/sin caches.
        with torch.no_grad():
            for name, param in module.named_parameters(recurse=True):
                if (
                    param.is_floating_point()
                    and "float8" not in str(param.dtype)
                    and "scale" not in name
                ):
                    param.data = param.data.to(dtype=dtype)
    module.eval()
    return module


def _first_floating_dtype(value: Any) -> torch.dtype | None:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        if "float8" not in str(value.dtype):
            return value.dtype
        return None
    if isinstance(value, dict):
        for v in value.values():
            dtype = _first_floating_dtype(v)
            if dtype is not None:
                return dtype
    if isinstance(value, (tuple, list)):
        for v in value:
            dtype = _first_floating_dtype(v)
            if dtype is not None:
                return dtype
    return None


def _clone_input_value(value: Any) -> Any:
    """Clone tensors in an input tree so in-place kernels cannot cross-contaminate runs."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_input_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_input_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_input_value(v) for k, v in value.items()}
    return value


def _clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: _clone_input_value(v) for k, v in inputs.items()}


def _balanced_cu_seqlens(
    total_tokens: int,
    batch: int,
    max_seqlen: int | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Build monotonic cu_seqlens preserving the recorded tensor shape."""
    batch = max(0, int(batch))
    max_seqlen = int(max_seqlen) if max_seqlen is not None else total_tokens
    max_seqlen = max(1, max_seqlen)
    if batch == 0:
        return torch.zeros(1, dtype=dtype, device=device), []
    if total_tokens > batch * max_seqlen:
        max_seqlen = math.ceil(total_tokens / batch)

    base, extra = divmod(int(total_tokens), batch)
    lengths = [base + (1 if i < extra else 0) for i in range(batch)]
    if any(length > max_seqlen for length in lengths):
        lengths = []
        remaining = int(total_tokens)
        for i in range(batch):
            slots_left = batch - i
            max_after = (slots_left - 1) * max_seqlen
            length = min(max_seqlen, max(0, remaining - max_after))
            lengths.append(length)
            remaining -= length
        if remaining > 0:
            lengths[-1] += remaining

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, dtype=dtype, device=device), lengths


def _prepare_cu_seqlens(inputs: dict[str, Any]) -> None:
    q_lengths: list[int] | None = None
    if isinstance(inputs.get("cu_seqlens_q"), torch.Tensor):
        cu = inputs["cu_seqlens_q"]
        q = inputs.get("q")
        if isinstance(q, torch.Tensor):
            total_q = int(q.shape[0])
            batch = max(0, int(cu.numel()) - 1)
            inputs["cu_seqlens_q"], q_lengths = _balanced_cu_seqlens(
                total_q,
                batch,
                int(inputs["max_seqlen_q"]) if "max_seqlen_q" in inputs else None,
                dtype=cu.dtype,
                device=cu.device,
            )

    if isinstance(inputs.get("cu_seqlens_k"), torch.Tensor):
        cu = inputs["cu_seqlens_k"]
        k = inputs.get("k")
        batch = max(0, int(cu.numel()) - 1)
        if (
            isinstance(k, torch.Tensor)
            and k.ndim == 4
            and isinstance(inputs.get("block_table"), torch.Tensor)
            and q_lengths is not None
            and len(q_lengths) == batch
        ):
            total_k = sum(q_lengths)
        elif isinstance(k, torch.Tensor):
            total_k = int(k.shape[0])
        elif q_lengths is not None:
            total_k = sum(q_lengths)
        else:
            total_k = batch
        inputs["cu_seqlens_k"], _ = _balanced_cu_seqlens(
            total_k,
            batch,
            int(inputs["max_seqlen_k"]) if "max_seqlen_k" in inputs else None,
            dtype=cu.dtype,
            device=cu.device,
        )


def _prepare_paged_attention_inputs(inputs: dict[str, Any]) -> None:
    block_table = inputs.get("block_table")
    if isinstance(block_table, torch.Tensor):
        num_blocks = 1
        for cache_name in ("k_cache", "k"):
            cache = inputs.get(cache_name)
            if isinstance(cache, torch.Tensor) and cache.ndim == 4:
                num_blocks = int(cache.shape[0])
                break
        inputs["block_table"] = (
            torch.arange(block_table.numel(), dtype=block_table.dtype, device=block_table.device)
            .reshape_as(block_table)
            .remainder(max(1, num_blocks))
        )

    cache_seqlens = inputs.get("cache_seqlens")
    if isinstance(cache_seqlens, torch.Tensor):
        max_seq_len = int(inputs.get("max_seq_len", 1))
        block_table = inputs.get("block_table")
        if isinstance(block_table, torch.Tensor):
            max_seq_len = min(max_seq_len, int(block_table.shape[1]) * 16)
        values = torch.full_like(cache_seqlens, max(1, max_seq_len))
        if cache_seqlens.numel() > 1:
            values -= torch.arange(
                cache_seqlens.numel(),
                dtype=cache_seqlens.dtype,
                device=cache_seqlens.device,
            ).remainder(min(max_seq_len, 7))
            values.clamp_(min=1)
        inputs["cache_seqlens"] = values


def _prepare_chunk_gla_inputs(inputs: dict[str, Any]) -> None:
    g = inputs.get("g")
    if isinstance(g, torch.Tensor):
        inputs["g"] = -torch.rand(g.shape, dtype=torch.float32, device=g.device).to(g.dtype) * 0.01

    cu = inputs.get("cu_seqlens")
    q = inputs.get("q")
    if isinstance(cu, torch.Tensor) and isinstance(q, torch.Tensor):
        total = int(q.shape[1] if q.ndim == 4 and q.shape[0] == 1 else q.shape[0] * q.shape[1])
        inputs["cu_seqlens"], _ = _balanced_cu_seqlens(
            total,
            max(0, int(cu.numel()) - 1),
            None,
            dtype=cu.dtype,
            device=cu.device,
        )


def _prepare_fp8_linear_inputs(inputs: dict[str, Any]) -> None:
    weight = inputs.get("weight_fp8")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return
    if str(weight.dtype) != "torch.float8_e4m3fn":
        return
    # The shape-only kernel runner should not require runtime FlashInfer
    # cubin compilation for tiny M; DeepGEMM covers the same contract here.
    os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")

    key = (str(weight.dtype), int(weight.shape[0]), int(weight.shape[1]), weight.device)
    cached = _FP8_WEIGHT_CACHE.get(key)
    if cached is None:
        from kb_nano.tasks.baseline.L1.fp8_linear import postprocess_fp8_weights

        n, k = int(weight.shape[0]), int(weight.shape[1])
        block = 128
        raw_scale = torch.ones(
            math.ceil(n / block),
            math.ceil(k / block),
            dtype=torch.float32,
            device=weight.device,
        ) * 0.02
        raw_weight = torch.randn(
            n, k, dtype=torch.bfloat16, device=weight.device,
        ).to(torch.float8_e4m3fn)
        cached = postprocess_fp8_weights(raw_weight, raw_scale)
        _FP8_WEIGHT_CACHE[key] = cached
    inputs["weight_fp8"], inputs["weight_scale_inv"] = cached


def _prepare_moe_grouped_gemm_inputs(inputs: dict[str, Any]) -> None:
    a = inputs.get("A")
    b = inputs.get("B")
    sorted_token_ids = inputs.get("sorted_token_ids")
    expert_ids = inputs.get("expert_ids")
    num_tokens_post_padded = inputs.get("num_tokens_post_padded")
    if not (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and isinstance(sorted_token_ids, torch.Tensor)
        and isinstance(expert_ids, torch.Tensor)
        and isinstance(num_tokens_post_padded, torch.Tensor)
    ):
        return

    top_k = int(inputs.get("top_k", 1))
    valid = int(a.shape[0]) * max(1, top_k)
    inputs["config"] = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 5,
    }
    block_size = int(inputs["config"]["BLOCK_SIZE_M"])
    padded = math.ceil(int(sorted_token_ids.numel()) / block_size) * block_size
    values = torch.full(
        (padded,),
        valid,
        dtype=sorted_token_ids.dtype,
        device=sorted_token_ids.device,
    )
    values[: min(valid, padded)] = torch.arange(
        min(valid, padded),
        dtype=sorted_token_ids.dtype,
        device=sorted_token_ids.device,
    )
    inputs["sorted_token_ids"] = values

    used_blocks = math.ceil(padded / block_size)
    expert_values = torch.arange(
        used_blocks,
        dtype=expert_ids.dtype,
        device=expert_ids.device,
    ).remainder(max(1, int(b.shape[0])))
    inputs["expert_ids"] = expert_values
    inputs["num_tokens_post_padded"] = torch.full_like(num_tokens_post_padded, padded)


def _round_up(value: int, align: int) -> int:
    return (int(value) + int(align) - 1) // int(align) * int(align)


def _gpt_oss_mxfp4_model_candidates(experts: int) -> list[str]:
    configured = os.environ.get("KB_NANO_GPT_OSS_MXFP4_MODEL")
    if configured:
        return [configured]
    if experts >= 128:
        return ["openai/gpt-oss-120b"]
    if experts <= 32:
        return ["openai/gpt-oss-20b"]
    return ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def _resolve_gpt_oss_mxfp4_paths(experts: int) -> list[Path]:
    configured = os.environ.get("KB_NANO_GPT_OSS_MXFP4_PATH")
    if configured:
        path = Path(configured).expanduser()
        return [path] if path.exists() else []

    paths: list[Path] = []
    try:
        from huggingface_hub import snapshot_download

        for model in _gpt_oss_mxfp4_model_candidates(experts):
            try:
                snapshot = snapshot_download(model, local_files_only=True)
            except Exception:
                continue
            paths.append(Path(snapshot))
    except Exception:
        pass
    return paths


def _safetensor_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".safetensors":
        return [path]
    if path.is_dir() and list(path.glob("*.safetensors")):
        return sorted(path.glob("*.safetensors"))
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        refs_main = path / "refs" / "main"
        if refs_main.is_file():
            snapshot = snapshots / refs_main.read_text().strip()
            if snapshot.is_dir():
                files = sorted(snapshot.glob("*.safetensors"))
                if files:
                    return files
        for snapshot in sorted(snapshots.iterdir()):
            if snapshot.is_dir():
                files = sorted(snapshot.glob("*.safetensors"))
                if files:
                    return files
    return []


def _reshape_gpt_oss_blocks(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])
    return tensor


def _try_load_gpt_oss_mxfp4_weights(
    experts: int,
    hidden_size: int,
    intermediate: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    paths = _resolve_gpt_oss_mxfp4_paths(experts)
    if not paths:
        return None

    expected = {
        "w1": [experts, 2 * intermediate, hidden_size // 2],
        "w1_scale": [experts, 2 * intermediate, hidden_size // _MXFP4_BLOCK_SIZE],
        "w1_bias": [experts, 2 * intermediate],
        "w2": [experts, hidden_size, intermediate // 2],
        "w2_scale": [experts, hidden_size, intermediate // _MXFP4_BLOCK_SIZE],
        "w2_bias": [experts, hidden_size],
    }
    key_suffix = {
        "w1": "mlp.experts.gate_up_proj_blocks",
        "w1_scale": "mlp.experts.gate_up_proj_scales",
        "w1_bias": "mlp.experts.gate_up_proj_bias",
        "w2": "mlp.experts.down_proj_blocks",
        "w2_scale": "mlp.experts.down_proj_scales",
        "w2_bias": "mlp.experts.down_proj_bias",
    }
    loaded: dict[str, torch.Tensor] = {}
    loaded_path: Path | None = None
    for path in paths:
        loaded.clear()
        for sf_file in _safetensor_files(path):
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for out_name, suffix in key_suffix.items():
                    if out_name in loaded:
                        continue
                    matches = sorted(k for k in keys if k.startswith("model.layers.") and k.endswith(suffix))
                    for key in matches:
                        tensor = f.get_tensor(key)
                        if out_name in ("w1", "w2"):
                            tensor = _reshape_gpt_oss_blocks(tensor)
                        if list(tensor.shape) == expected[out_name]:
                            loaded[out_name] = tensor
                            break
            if len(loaded) == len(expected):
                loaded_path = path
                break
        if loaded_path is not None:
            break

    if len(loaded) != len(expected):
        return None

    return {
        "w1": loaded["w1"].to(device=device, dtype=torch.uint8),
        "w1_scale": loaded["w1_scale"].to(device=device, dtype=torch.uint8),
        "w1_bias": loaded["w1_bias"].to(device=device, dtype=torch.float32),
        "w2": loaded["w2"].to(device=device, dtype=torch.uint8),
        "w2_scale": loaded["w2_scale"].to(device=device, dtype=torch.uint8),
        "w2_bias": loaded["w2_bias"].to(device=device, dtype=torch.float32),
        "source": f"checkpoint:{loaded_path}",
    }


def _generate_gpt_oss_mxfp4_weights(
    experts: int,
    hidden_size: int,
    intermediate: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260506)

    def packed(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8, device="cpu").to(device)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        # E8M0 value 127 corresponds to scale 1.0 and keeps random packed FP4 finite.
        return torch.full(shape, 127, dtype=torch.uint8, device=device)

    return {
        "w1": packed((experts, 2 * intermediate, hidden_size // 2)),
        "w1_scale": scales((experts, 2 * intermediate, hidden_size // _MXFP4_BLOCK_SIZE)),
        "w1_bias": torch.zeros((experts, 2 * intermediate), dtype=torch.float32, device=device),
        "w2": packed((experts, hidden_size, intermediate // 2)),
        "w2_scale": scales((experts, hidden_size, intermediate // _MXFP4_BLOCK_SIZE)),
        "w2_bias": torch.zeros((experts, hidden_size), dtype=torch.float32, device=device),
        "source": "synthetic",
    }


def _get_mxfp4_raw_weights(
    experts: int,
    hidden_size: int,
    intermediate: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    source_path = os.environ.get("KB_NANO_GPT_OSS_MXFP4_PATH") or os.environ.get(
        "KB_NANO_GPT_OSS_MXFP4_MODEL",
        f"auto:{experts}",
    )
    key = (experts, hidden_size, intermediate, device, source_path)
    cached = _MXFP4_RAW_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    loaded = _try_load_gpt_oss_mxfp4_weights(experts, hidden_size, intermediate, device)
    cached = loaded if loaded is not None else _generate_gpt_oss_mxfp4_weights(
        experts,
        hidden_size,
        intermediate,
        device,
    )
    _MXFP4_RAW_WEIGHT_CACHE[key] = cached
    return cached


def _prepare_mxfp4_moe_inputs(inputs: dict[str, Any]) -> None:
    hidden = inputs.get("hidden_states")
    gating = inputs.get("gating_output")
    if not isinstance(hidden, torch.Tensor) or not isinstance(gating, torch.Tensor):
        return
    if "_mxfp4_raw" in inputs or ("w1" in inputs and "w2" in inputs and "quant_config" in inputs):
        return

    experts = int(gating.shape[-1])
    hidden_size = int(hidden.shape[-1])
    intermediate = int(inputs.pop("intermediate_size", hidden_size))
    intermediate = _round_up(intermediate, 64)
    inputs["_mxfp4_raw"] = _get_mxfp4_raw_weights(
        experts,
        hidden_size,
        intermediate,
        hidden.device,
    )


def _prepare_fused_experts_inputs(inputs: dict[str, Any]) -> None:
    topk_ids = inputs.get("topk_ids")
    topk_weights = inputs.get("topk_weights")
    num_experts = int(inputs.get("num_experts", 0))
    if isinstance(topk_ids, torch.Tensor) and num_experts > 0:
        inputs["topk_ids"] = topk_ids.remainder(num_experts).to(torch.int32)
    if isinstance(topk_weights, torch.Tensor):
        inputs["topk_weights"] = torch.softmax(topk_weights.float(), dim=-1)

    if bool(inputs.get("use_fp8_w8a8", False)):
        for key in ("w13_scale", "w13_scale_dg", "w2_scale", "w2_scale_dg"):
            scale = inputs.get(key)
            if isinstance(scale, torch.Tensor):
                inputs[key] = torch.full_like(scale.float(), 0.005)


def _prepare_inputs_for_target(
    target_name: str,
    inputs: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Fill deterministic defaults for trace-only scalar scenarios."""
    try:
        from kb_nano.infra.context import set_context

        set_context(False)
    except Exception:
        pass

    _prepare_cu_seqlens(inputs)
    _prepare_paged_attention_inputs(inputs)

    if target_name == "mrope_input_positions" and "input_tokens" not in inputs:
        offsets = []
        for key in ("image_offsets", "video_offsets"):
            value = inputs.get(key)
            if isinstance(value, list):
                offsets.extend(int(v) for v in value)
        seq_len = (max(offsets) + 16) if offsets else 16
        inputs["input_tokens"] = [0] * seq_len

    if target_name == "vision_rotary_emb":
        sms = int(inputs.get("spatial_merge_size", 2))
        inputs.setdefault("grid_thw_list", [[1, sms, sms]])
        inputs.setdefault("dtype", torch.bfloat16)
        inputs.setdefault("device", torch.device(device))

    if target_name == "vision_pos_embed_interpolate":
        inputs.setdefault("grid_thw_list", [[1, 2, 2]])
        inputs.setdefault("dtype", torch.bfloat16)
        inputs.setdefault("device", torch.device(device))

    if target_name == "yolov10_concat":
        inputs.setdefault(
            "xs",
            [
                torch.randn((1, 8, 4, 4), dtype=torch.bfloat16, device=device),
                torch.randn((1, 8, 4, 4), dtype=torch.bfloat16, device=device),
            ],
        )

    if target_name == "oasis_patch_embed" and isinstance(inputs.get("x"), torch.Tensor):
        inputs["x"] = inputs["x"].to(torch.bfloat16)

    if target_name in ("attention", "attention_impl"):
        try:
            from kb_nano.infra.context import set_context

            tensor = inputs.get("hidden_states", inputs.get("query"))
            if isinstance(tensor, torch.Tensor):
                n_tokens = int(tensor.shape[0])
                cu = torch.tensor([0, n_tokens], dtype=torch.int32, device=tensor.device)
                set_context(
                    True,
                    cu_seqlens_q=cu,
                    cu_seqlens_k=cu,
                    max_seqlen_q=n_tokens,
                    max_seqlen_k=n_tokens,
                )
        except Exception:
            pass

    if target_name == "gla_attention":
        cu = inputs.get("cu_seqlens")
        hidden_states = inputs.get("hidden_states")
        if isinstance(cu, torch.Tensor) and isinstance(hidden_states, torch.Tensor):
            total_tokens = int(
                hidden_states.shape[1]
                if hidden_states.ndim == 3 and hidden_states.shape[0] == 1
                else hidden_states.shape[0] * hidden_states.shape[1]
            )
            inputs["cu_seqlens"], _ = _balanced_cu_seqlens(
                total_tokens,
                max(0, int(cu.numel()) - 1),
                None,
                dtype=cu.dtype,
                device=cu.device,
            )

    if target_name == "store_kvcache" and isinstance(inputs.get("slot_mapping"), torch.Tensor):
        slot_mapping = inputs["slot_mapping"]
        inputs["slot_mapping"] = torch.arange(
            slot_mapping.numel(),
            dtype=slot_mapping.dtype,
            device=slot_mapping.device,
        )

    if target_name == "chunk_gla":
        _prepare_chunk_gla_inputs(inputs)

    if target_name == "fp8_linear":
        _prepare_fp8_linear_inputs(inputs)

    if target_name == "parallel_linear":
        os.environ["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] = "0"

    if target_name == "moe_grouped_gemm":
        _prepare_moe_grouped_gemm_inputs(inputs)

    if target_name == "fused_experts":
        _prepare_fused_experts_inputs(inputs)

    if target_name == "mxfp4_moe":
        _prepare_mxfp4_moe_inputs(inputs)

    return inputs


def _canonicalize_output_for_target(
    target_name: str,
    output: Any,
    inputs: dict[str, Any],
) -> Any:
    if target_name == "moe_align" and isinstance(output, (tuple, list)) and len(output) == 3:
        sorted_token_ids, expert_ids, num_tokens_post_padded = output
        if not isinstance(num_tokens_post_padded, torch.Tensor):
            return output
        valid = int(num_tokens_post_padded.reshape(-1)[0].item())
        block_size = int(inputs.get("block_size", 1))
        if isinstance(sorted_token_ids, torch.Tensor) and isinstance(expert_ids, torch.Tensor):
            sorted_token_ids = sorted_token_ids[:valid]
            expert_ids = expert_ids[: math.ceil(valid / max(1, block_size))]
            if sorted_token_ids.numel() > 0:
                expanded_experts = expert_ids.repeat_interleave(max(1, block_size))[: sorted_token_ids.numel()]
                token_range = int(sorted_token_ids.max().item()) + 1
                token_range = max(token_range, 1)
                order = torch.argsort(expanded_experts.to(torch.int64) * token_range + sorted_token_ids.to(torch.int64))
                sorted_token_ids = sorted_token_ids[order]
                expert_ids = expanded_experts[order]
        return type(output)((sorted_token_ids, expert_ids, num_tokens_post_padded))
    return output


def _prepare_mxfp4_inputs_for_module(module: nn.Module, inputs: dict[str, Any]) -> dict[str, Any]:
    raw = inputs.get("_mxfp4_raw")
    if not isinstance(raw, dict):
        return _clone_inputs(inputs)

    hidden = inputs.get("hidden_states")
    gating = inputs.get("gating_output")
    if not isinstance(hidden, torch.Tensor) or not isinstance(gating, torch.Tensor):
        return _clone_inputs(inputs)

    experts = int(gating.shape[-1])
    hidden_size = int(hidden.shape[-1])
    intermediate = int(raw["w2"].shape[-1]) * 2
    source = str(raw.get("source", ""))
    key = (type(module), experts, hidden_size, intermediate, hidden.device, source)
    prepared = _MXFP4_PREPARED_WEIGHT_CACHE.get(key)
    if prepared is None:
        w1, w1_precision = module.prepare_weight(raw["w1"], raw["w1_scale"])
        w2, w2_precision = module.prepare_weight(raw["w2"], raw["w2_scale"])
        quant_config = module.make_quant_config(
            w1_precision=w1_precision,
            w2_precision=w2_precision,
            w1_bias=raw["w1_bias"],
            w2_bias=raw["w2_bias"],
        )
        prepared = (w1, w2, quant_config)
        _MXFP4_PREPARED_WEIGHT_CACHE[key] = prepared

    result = {
        k: _clone_input_value(v)
        for k, v in inputs.items()
        if not k.startswith("_mxfp4_")
    }
    result["w1"], result["w2"], result["quant_config"] = prepared
    return result


def _prepare_inputs_for_module(
    target_name: str,
    module: nn.Module,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    if target_name == "mxfp4_moe":
        return _prepare_mxfp4_inputs_for_module(module, inputs)
    return _clone_inputs(inputs)


def _initialize_parameters(module: nn.Module) -> None:
    """Replace torch.empty() parameters with finite deterministic values."""
    generator = torch.Generator(device="cpu").manual_seed(20260506)
    with torch.no_grad():
        for name, param in module.named_parameters(recurse=True):
            if not param.is_floating_point():
                param.zero_()
                continue
            if "scale" in name and "float8" not in str(param.dtype):
                param.fill_(0.005)
                continue
            values = torch.randn(
                param.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device=param.device)
            values = values * 0.0005
            param.copy_(values.to(dtype=param.dtype))


def _contains_cuda_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, dict):
        return any(_contains_cuda_tensor(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_cuda_tensor(v) for v in value)
    return False


def _synchronize_if_cuda(*values: Any) -> None:
    if any(_contains_cuda_tensor(v) for v in values):
        torch.cuda.synchronize()


def _time_forward(
    module: nn.Module,
    inputs: dict[str, Any],
    num_warmup: int,
    num_runs: int,
) -> tuple[Any, float]:
    """Warmup + time forward() calls. Returns (output, median_ms)."""
    tensor_inputs = {
        k: v for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }
    scalar_inputs = {
        k: v for k, v in inputs.items()
        if not isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        for _ in range(num_warmup):
            module(**tensor_inputs, **scalar_inputs)

        _synchronize_if_cuda(tensor_inputs)
        times = []
        output = None
        for _ in range(num_runs):
            start = time.perf_counter()
            output = module(**tensor_inputs, **scalar_inputs)
            _synchronize_if_cuda(tensor_inputs, output)
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    median_ms = times[len(times) // 2]
    if output is None:
        output = {k: v for k, v in tensor_inputs.items()}
    return output, median_ms


def _run_forward_once(module: nn.Module, inputs: dict[str, Any]) -> Any:
    tensor_inputs = {
        k: v for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }
    scalar_inputs = {
        k: v for k, v in inputs.items()
        if not isinstance(v, torch.Tensor)
    }
    with torch.no_grad():
        output = module(**tensor_inputs, **scalar_inputs)
        _synchronize_if_cuda(tensor_inputs, output)
    return output


def _tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return (atol, rtol) for tolerance-normalized correctness."""
    dtype_name = str(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return _LOW_PRECISION_ATOL, _LOW_PRECISION_RTOL
    if "float8" in dtype_name:
        return _FP8_ATOL, _FP8_RTOL
    return _FP32_ATOL, _FP32_RTOL


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _is_fp8_tensor(tensor: Any) -> bool:
    return isinstance(tensor, torch.Tensor) and "float8" in str(tensor.dtype)


def _fp8_rtol(dtype: torch.dtype) -> float:
    if "e5m2" in str(dtype):
        return 0.25
    return 0.125


def _expand_fp8_scale(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int = _FP8_GROUP_SIZE,
) -> torch.Tensor | None:
    """Broadcast FP8 per-group/per-block scales to the FP8 tensor shape."""
    if not isinstance(scale, torch.Tensor) or not scale.is_floating_point():
        return None

    shape = tuple(fp8.shape)
    scale_shape = tuple(scale.shape)
    per_group_shape = (*shape[:-1], _ceil_div(shape[-1], group_size))
    if scale_shape == per_group_shape:
        return scale.float().repeat_interleave(group_size, dim=-1)[..., :shape[-1]]

    if fp8.ndim >= 2:
        per_block_shape = (
            *shape[:-2],
            _ceil_div(shape[-2], group_size),
            _ceil_div(shape[-1], group_size),
        )
        if scale_shape == per_block_shape:
            expanded = scale.float().repeat_interleave(group_size, dim=-2)
            expanded = expanded.repeat_interleave(group_size, dim=-1)
            return expanded[..., :shape[-2], :shape[-1]]

    return None


def _compare_fp8_scaled_outputs(
    baseline_fp8: torch.Tensor,
    baseline_scale: torch.Tensor,
    candidate_fp8: torch.Tensor,
    candidate_scale: torch.Tensor,
) -> tuple[bool, float, float]:
    """Compare FP8 tensors in dequantized value space using local scales."""
    if baseline_fp8.shape != candidate_fp8.shape:
        return False, float("inf"), float("inf")

    baseline_scale_expanded = _expand_fp8_scale(baseline_fp8, baseline_scale)
    candidate_scale_expanded = _expand_fp8_scale(candidate_fp8, candidate_scale)
    if baseline_scale_expanded is None or candidate_scale_expanded is None:
        return False, float("inf"), float("inf")

    baseline = baseline_fp8.float() * baseline_scale_expanded
    candidate = candidate_fp8.float() * candidate_scale_expanded
    if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        return False, float("inf"), float("inf")

    diff = (baseline - candidate).abs()
    mean_diff = diff.mean().item()
    atol = 0.5 * baseline_scale_expanded.abs().clamp_min(1e-12)
    tolerance = atol + _fp8_rtol(baseline_fp8.dtype) * baseline.abs()
    max_error_ratio = (diff / tolerance).max().item()
    passed = max_error_ratio <= 1.0
    return passed, max_error_ratio, mean_diff


def _compare_outputs(baseline_out: Any, candidate_out: Any) -> tuple[bool, float, float]:
    """Compare outputs: return (pass, max_error_ratio, mean_abs_diff)."""
    if isinstance(baseline_out, torch.Tensor) and isinstance(candidate_out, torch.Tensor):
        if baseline_out.shape != candidate_out.shape:
            return False, float("inf"), float("inf")

        baseline = baseline_out.float()
        candidate = candidate_out.float()
        if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
            return False, float("inf"), float("inf")

        diff = (baseline - candidate).abs()
        mean_diff = diff.mean().item()
        atol, rtol = _tolerances_for_dtype(baseline_out.dtype)
        tolerance = atol + rtol * baseline.abs()
        max_error_ratio = (diff / tolerance).max().item()
        passed = max_error_ratio <= 1.0
        return passed, max_error_ratio, mean_diff

    if isinstance(baseline_out, (tuple, list)) and isinstance(candidate_out, (tuple, list)):
        if len(baseline_out) != len(candidate_out):
            return False, float("inf"), float("inf")
        all_pass = True
        max_error_ratio = 0.0
        total_diff = 0.0
        count = 0
        i = 0
        while i < len(baseline_out):
            b = baseline_out[i]
            c = candidate_out[i]
            if (
                i + 1 < len(baseline_out)
                and _is_fp8_tensor(b)
                and _is_fp8_tensor(c)
                and isinstance(baseline_out[i + 1], torch.Tensor)
                and isinstance(candidate_out[i + 1], torch.Tensor)
            ):
                baseline_scale = baseline_out[i + 1]
                candidate_scale = candidate_out[i + 1]
                if (
                    _expand_fp8_scale(b, baseline_scale) is not None
                    and _expand_fp8_scale(c, candidate_scale) is not None
                ):
                    p, ratio, d = _compare_fp8_scaled_outputs(
                        b, baseline_scale, c, candidate_scale,
                    )
                    all_pass = all_pass and p
                    max_error_ratio = max(max_error_ratio, ratio)
                    total_diff += d
                    count += 1
                    i += 2
                    continue

            if isinstance(b, torch.Tensor) and isinstance(c, torch.Tensor):
                p, ratio, d = _compare_outputs(b, c)
                all_pass = all_pass and p
                max_error_ratio = max(max_error_ratio, ratio)
                total_diff += d
                count += 1
            i += 1
        mean_diff = total_diff / count if count > 0 else 0.0
        return all_pass, max_error_ratio, mean_diff

    if isinstance(baseline_out, dict) and isinstance(candidate_out, dict):
        if set(baseline_out) != set(candidate_out):
            return False, float("inf"), float("inf")
        all_pass = True
        max_error_ratio = 0.0
        total_diff = 0.0
        count = 0
        for key in sorted(baseline_out):
            b = baseline_out[key]
            c = candidate_out[key]
            p, ratio, d = _compare_outputs(b, c)
            all_pass = all_pass and p
            max_error_ratio = max(max_error_ratio, ratio)
            total_diff += d
            count += 1
        mean_diff = total_diff / count if count > 0 else 0.0
        return all_pass, max_error_ratio, mean_diff

    return True, 0.0, 0.0


def _merge_correctness(
    output_check: tuple[bool, float, float],
    input_check: tuple[bool, float, float],
) -> tuple[bool, float, float]:
    output_correct, output_ratio, output_diff = output_check
    input_correct, input_ratio, input_diff = input_check
    correct = output_correct and input_correct
    max_error_ratio = max(output_ratio, input_ratio)
    if output_diff == 0.0:
        mean_diff = input_diff
    elif input_diff == 0.0:
        mean_diff = output_diff
    else:
        mean_diff = 0.5 * (output_diff + input_diff)
    return correct, max_error_ratio, mean_diff


def run_kernel_benchmark(
    target_name: str,
    scenarios: list[str] | None = None,
    models: list[str] | None = None,
    tp: list[int] | None = None,
    category: str | None = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    pytorch_reference: bool = False,
    validation_mode: str = "candidate",
) -> OperatorResult:
    """Run isolated kernel benchmark for a single operator.

    For each matching scenario in the InputRegistry:
    1. Instantiate baseline and candidate with init_args
    2. Copy baseline weights to candidate (via load_state_dict)
    3. Prepare inputs (random or golden)
    4. Warmup both
    5. Time both (median of num_runs)
    6. Compare outputs: max error ratio pass/fail, mean abs diff

    The candidate implementation is auto-discovered from
    tasks/candidate/L{level}/{target_name}.py.

    Args:
        target_name: Operator name (e.g. 'rms_norm').
        scenarios: Filter by scenario name patterns.
        models: Filter by model key prefix.
        tp: Filter by TP degrees.
        category: Filter by category (not yet used).
        num_warmup: Warmup iterations.
        num_runs: Timed iterations for median.
        device: Device for tensors.

    Returns:
        OperatorResult with per-scenario correctness and speedup.
    """
    target = get(target_name)

    if pytorch_reference:
        validation_mode = "pytorch_reference"

    if validation_mode == "baseline_identity":
        user_impl = target.target_cls
    elif validation_mode == "pytorch_reference":
        user_impl = load_reference(target_name)
    else:
        user_impl = load_candidate(target_name)

    if user_impl is None:
        impl_kind = "PyTorch reference" if validation_mode == "pytorch_reference" else "candidate kernel"
        impl_dir = "reference" if validation_mode == "pytorch_reference" else "candidate"
        raise ValueError(
            f"No {impl_kind} found for {target_name!r}. "
            f"Place implementation in tasks/{impl_dir}/L{target.level}/{target_name}.py"
        )

    registry = _get_registry()
    all_scenarios = registry.scenarios(
        target_name, models=models, tp=tp, category=category,
    )

    if scenarios:
        all_scenarios = [
            s for s in all_scenarios
            if any(pat in s.name for pat in scenarios)
        ]

    if not all_scenarios:
        print(f"  WARNING: No scenarios found for {target_name} in InputRegistry.")
        return OperatorResult(
            target=target_name,
            level=target.level,
            candidate_path=(
                _find_reference_path(target_name, target.level)
                if pytorch_reference
                else _find_candidate_path(target_name, target.level)
            ),
        )

    if validation_mode == "baseline_identity":
        candidate_path = f"tasks/baseline/L{target.level}/{target_name}.py"
    elif validation_mode == "pytorch_reference":
        candidate_path = _find_reference_path(target_name, target.level)
    else:
        candidate_path = _find_candidate_path(target_name, target.level)
    scenario_results: list[ScenarioResult] = []

    for scenario in all_scenarios:
        try:
            inputs = registry.get_inputs(target_name, scenario.name, device=device)
            inputs = _prepare_inputs_for_target(target_name, inputs, device)
            input_dtype = _first_floating_dtype(inputs)

            baseline_mod = _instantiate_module(
                target.target_cls, scenario.init_args, inputs, device, dtype=input_dtype,
            )
            candidate_mod = _instantiate_module(
                user_impl, scenario.init_args, inputs, device, dtype=input_dtype,
            )
            _initialize_parameters(baseline_mod)

            if hasattr(baseline_mod, "state_dict") and len(baseline_mod.state_dict()) > 0:
                try:
                    candidate_mod.load_state_dict(baseline_mod.state_dict(), strict=False)
                except Exception:
                    pass

            timing_warmup = 0 if validation_mode == "candidate_smoke" else num_warmup
            timing_runs = 1 if validation_mode == "candidate_smoke" else num_runs

            baseline_check_inputs = _prepare_inputs_for_module(
                target_name, baseline_mod, inputs,
            )
            candidate_check_inputs = _prepare_inputs_for_module(
                target_name, candidate_mod, inputs,
            )
            baseline_out = _run_forward_once(baseline_mod, baseline_check_inputs)
            candidate_out = _run_forward_once(candidate_mod, candidate_check_inputs)
            baseline_out = _canonicalize_output_for_target(
                target_name, baseline_out, baseline_check_inputs,
            )
            candidate_out = _canonicalize_output_for_target(
                target_name, candidate_out, candidate_check_inputs,
            )

            correct, max_error_ratio, mean_diff = _merge_correctness(
                _compare_outputs(baseline_out, candidate_out),
                _compare_outputs(baseline_check_inputs, candidate_check_inputs),
            )

            _, baseline_ms = _time_forward(
                baseline_mod,
                _prepare_inputs_for_module(target_name, baseline_mod, inputs),
                timing_warmup,
                timing_runs,
            )
            _, candidate_ms = _time_forward(
                candidate_mod,
                _prepare_inputs_for_module(target_name, candidate_mod, inputs),
                timing_warmup,
                timing_runs,
            )
            speedup = baseline_ms / candidate_ms if candidate_ms > 0 else float("inf")
            classification = (
                "harness_validation_passed"
                if validation_mode in ("baseline_identity", "pytorch_reference")
                and correct
                else "candidate_correct_and_timed"
                if correct
                else "candidate_correctness_failure"
            )

            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=correct,
                max_error_ratio=max_error_ratio,
                mean_abs_diff=mean_diff,
                baseline_ms=baseline_ms,
                candidate_ms=candidate_ms,
                speedup=speedup,
                failure_reason=None if correct else "output_mismatch",
                classification=classification,
            ))

        except Exception as e:
            failure_reason = _short_exception(e)
            print(f"  ERROR in scenario {scenario.name}: {failure_reason}")
            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=False,
                max_error_ratio=float("inf"),
                mean_abs_diff=float("inf"),
                baseline_ms=0.0,
                candidate_ms=0.0,
                speedup=0.0,
                failure_reason=failure_reason,
                classification="harness_or_candidate_exception",
            ))

        finally:
            for v in list(locals().values()):
                if isinstance(v, nn.Module):
                    del v

    op_result = OperatorResult(
        target=target_name,
        level=target.level,
        candidate_path=candidate_path,
        scenarios=scenario_results,
    )
    op_result.compute_aggregates()
    return op_result


def run_all_kernel_benchmarks(
    models: list[str] | None = None,
    tp: list[int] | None = None,
    category: str | None = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    pytorch_reference: bool = False,
    validation_mode: str = "candidate",
) -> KernelBenchResult:
    """Run kernel benchmarks for all operators that have candidate implementations.

    Discovers all candidate kernels and runs isolated benchmarks for each.
    """
    from kb_nano.infra.kernel_swapper import discover_candidates

    if pytorch_reference:
        validation_mode = "pytorch_reference"

    candidates = discover_references() if validation_mode == "pytorch_reference" else discover_candidates()
    if not candidates:
        if validation_mode == "pytorch_reference":
            print("No PyTorch references found in tasks/reference/.")
        else:
            print("No candidate kernels found in tasks/candidate/.")
        result = KernelBenchResult()
        result.compute_aggregates()
        return result

    operators: list[OperatorResult] = []
    for target, _ in candidates:
        label = (
            "baseline identity"
            if validation_mode == "baseline_identity"
            else "PyTorch reference"
            if validation_mode == "pytorch_reference"
            else "candidate smoke"
            if validation_mode == "candidate_smoke"
            else "candidate"
        )
        print(f"\n  Benchmarking {target.name} (L{target.level}, {label})...")
        op_result = run_kernel_benchmark(
            target.name,
            models=models,
            tp=tp,
            category=category,
            num_warmup=num_warmup,
            num_runs=num_runs,
            device=device,
            pytorch_reference=pytorch_reference,
            validation_mode=validation_mode,
        )
        operators.append(op_result)

    result = KernelBenchResult(operators=operators)
    result.compute_aggregates()
    return result
