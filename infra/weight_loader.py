"""
Weight loader for Llama 3.1, Llama 4, Mixtral, DeepSeek V3.2, Qwen2-VL,
and Qwen3-VL with tensor parallelism.

Loads weights from HuggingFace safetensors and distributes them
across TP shards using the weight_loader callbacks on each parameter.
"""

from __future__ import annotations

import os
import re
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

try:
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
    _HAS_FASTSAFETENSORS = True
except ImportError:
    _HAS_FASTSAFETENSORS = False

from .tp import _tp_size
from ..tasks.baseline.L1.fp8_linear import Fp8Linear, postprocess_fp8_weights, postprocess_fp8_weights_batched
from ..tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from ..tasks.baseline.L4.llama4 import Llama4Config, Llama4ForCausalLM
from ..tasks.baseline.L4.mixtral import MixtralConfig, MixtralForCausalLM
from ..tasks.baseline.L4.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from ..tasks.baseline.L4.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from ..tasks.baseline.L4.deepseek import DeepSeekV3Config, DeepSeekV3ForCausalLM


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def download_model(model_name: str) -> str:
    return snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )


_EXPERT_RE = re.compile(
    r"(.+\.block_sparse_moe)\.experts\.(\d+)\.(w[123])\.weight"
)

_DEEPSEEK_EXPERT_RE = re.compile(
    r"(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)"
)

_DEEPSEEK_SHARED_EXPERT_RE = re.compile(
    r"(.+\.mlp)\.shared_experts\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)"
)


# Qwen2-VL weight name remapping: checkpoint -> model parameter
# Checkpoint uses `model.` prefix for language, `visual.` for vision.
# Our model uses `model.` for language backbone and `visual.` for vision.
_QWEN2_VL_PREFIX_MAP = {
    "lm_head.": "lm_head.",
    "model.": "model.",
    "visual.": "visual.",
}

# Qwen3-VL weight name remapping: checkpoint uses `model.language_model.`
# and `model.visual.` prefixes.
_QWEN3_VL_PREFIX_MAP = {
    "model.language_model.": "model.",
    "model.visual.": "visual.",
    "lm_head.": "lm_head.",
}

# Vision encoder merged QKV pattern
_VISION_QKV_RE = re.compile(r"(.+\.attn)\.qkv\.(weight|bias)")

# Qwen3-VL vision MLP uses linear_fc1/linear_fc2 -> fc1/fc2
_QWEN3_VISION_MLP_RE = re.compile(r"(.+\.mlp)\.linear_fc([12])\.(weight|bias)")

# Qwen3-VL merger uses linear_fc1/linear_fc2 -> fc1/fc2
_QWEN3_MERGER_FC_RE = re.compile(r"(.+)\.(linear_fc1|linear_fc2)\.(weight|bias)")

# Qwen2-VL merger remapping: ln_q -> norm, mlp.0 -> fc1, mlp.2 -> fc2
_QWEN2_MERGER_RE = re.compile(r"(visual\.merger)\.(ln_q|mlp\.0|mlp\.2)\.(weight|bias)")

# Qwen3-VL learned pos embed: visual.pos_embed.weight -> visual.pos_embed_interp.pos_embed
_VISION_POS_EMBED_RE = re.compile(r"visual\.pos_embed\.weight$")

# L1 wrapper nesting: patch_embed.proj.X -> patch_embed.proj.conv.X
_VISION_PATCH_EMBED_RE = re.compile(r"(visual\.patch_embed\.proj)\.(weight|bias)")
# L1 wrapper nesting: *.norm1.X / *.norm2.X -> *.norm1.norm.X / *.norm2.norm.X (VisionBlock)
_VISION_BLOCK_NORM_RE = re.compile(r"(visual\.blocks\.\d+\.norm[12])\.(weight|bias)")
# L1 wrapper nesting: *.merger*.norm.X -> *.merger*.norm.norm.X (VisionPatchMerger)
_VISION_MERGER_NORM_RE = re.compile(r"(visual\.(?:merger|deepstack_merger_list\.\d+)\.norm)\.(weight|bias)")


# Llama4 fused expert weight patterns
_LLAMA4_FUSED_EXPERT_RE = re.compile(
    r"(.+\.feed_forward)\.experts\.(gate_up_proj|down_proj)"
)


def _permute_qk_for_rotary(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Permute Q/K weights from interleaved to contiguous layout for rotary."""
    f_out, f_in = weight.shape
    return (
        weight.view(n_heads, f_out // n_heads // 2, 2, f_in)
        .transpose(1, 2)
        .reshape(f_out, f_in)
    )


def _remap_qwen2_vl_name(name: str) -> str:
    """Remap Qwen2-VL checkpoint weight names to our model parameter names."""
    for prefix, replacement in _QWEN2_VL_PREFIX_MAP.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):]
    return name


def _remap_qwen3_vl_name(name: str) -> str:
    """Remap Qwen3-VL checkpoint weight names to our model parameter names."""
    for prefix, replacement in _QWEN3_VL_PREFIX_MAP.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):]
    return name


def _load_vision_qkv(model, param_prefix: str, loaded_weight: torch.Tensor,
                      wb: str) -> int:
    """Load merged vision QKV weight/bias into our QKVParallelLinear.

    The checkpoint stores Q, K, V concatenated as a single tensor.
    We split and load as separate shards.
    """
    total_dim = loaded_weight.shape[0]
    per_head = total_dim // 3
    q, k, v = loaded_weight.narrow(0, 0, per_head), \
               loaded_weight.narrow(0, per_head, per_head), \
               loaded_weight.narrow(0, 2 * per_head, per_head)
    param_name = f"{param_prefix}.qkv.{wb}"
    try:
        param = model.get_parameter(param_name)
    except AttributeError:
        return 0
    wl = param.weight_loader
    wl(param, q, "q")
    wl(param, k, "k")
    wl(param, v, "v")
    return 1


_WEIGHT_SCALE_INV_RE = re.compile(r"(.+)\.weight_scale_inv$")


def _weights_iterator(safetensor_files, use_fastsafetensors=True):
    """Iterate over (weight_name, tensor) from safetensor files.

    Uses fastsafetensors (GPU Direct Storage) when available for fast
    GPU-direct loading. Falls back to safetensors CPU loading otherwise.
    Matches vllm's fastsafetensors_weights_iterator / safetensors_weights_iterator.
    """
    if use_fastsafetensors and _HAS_FASTSAFETENSORS:
        import torch.distributed as dist
        if dist.is_initialized():
            pg = dist.group.WORLD
        else:
            pg = SingleGroup()
        device = torch.device(f"cuda:{pg.rank()}")
        batch_size = pg.size()
        file_batches = [
            safetensor_files[i:i + batch_size]
            for i in range(0, len(safetensor_files), batch_size)
        ]
        nogds = False
        for f_list in file_batches:
            loader = SafeTensorsFileLoader(pg, device, nogds=nogds)
            rank_file_map = {i: [f] for i, f in enumerate(f_list)}
            loader.add_filenames(rank_file_map)
            try:
                try:
                    fb = loader.copy_files_to_device()
                except RuntimeError as e:
                    if "gds" not in str(e):
                        raise
                    loader.close()
                    nogds = True
                    loader = SafeTensorsFileLoader(pg, device, nogds=nogds)
                    loader.add_filenames(rank_file_map)
                    fb = loader.copy_files_to_device()
                try:
                    for k in list(fb.key_to_rank_lidx.keys()):
                        yield k, fb.get_tensor(k)
                finally:
                    fb.close()
            finally:
                loader.close()
    else:
        for sf_file in safetensor_files:
            with safe_open(sf_file, "pt", "cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)


def load_weights(model, model_path: str, model_type: str = "llama") -> None:
    """Load weights with support for packed modules, MoE experts, vision
    encoder QKV, FP8 weight_scale_inv, and TP sharding.

    Uses fastsafetensors (GPU Direct Storage) when available to load tensors
    directly to GPU, matching vllm's fastsafetensors_weights_iterator.
    """
    packed = getattr(model, "packed_modules_mapping", {})
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    is_qwen2_vl = model_type == "qwen2_vl"
    is_qwen3_vl = model_type == "qwen3_vl"
    is_qwen_vl = is_qwen2_vl or is_qwen3_vl
    is_llama4 = model_type == "llama4"
    if is_llama4:
        llama4_config = model.config

    import time as _time
    _t_load = _time.perf_counter()
    n_files = len(safetensor_files)
    use_fast = _HAS_FASTSAFETENSORS
    print(f"  Loading weights from {n_files} safetensors file(s)"
          f" (fastsafetensors={'yes' if use_fast else 'no'})...", flush=True)
    loaded = 0
    _last_report = _t_load
    for weight_name, tensor in _weights_iterator(safetensor_files, use_fastsafetensors=use_fast):
        now = _time.perf_counter()
        if now - _last_report > 5.0:
            print(f"    {loaded} shards loaded ({now - _t_load:.1f}s)", flush=True)
            _last_report = now

        if is_qwen2_vl:
            mapped_name = _remap_qwen2_vl_name(weight_name)
        elif is_qwen3_vl:
            mapped_name = _remap_qwen3_vl_name(weight_name)
        else:
            mapped_name = weight_name

        if model_type == "deepseek_v3":
            mapped_name = mapped_name.replace(
                ".shared_experts.", ".shared_expert.")
            mapped_name = mapped_name.replace(
                ".mlp.gate.weight", ".mlp.gate_weight")

        if is_llama4:
            if not mapped_name.startswith("language_model."):
                continue
            mapped_name = mapped_name[len("language_model."):]

            m_fused = _LLAMA4_FUSED_EXPERT_RE.match(mapped_name)
            if m_fused:
                prefix_part, proj = m_fused.groups()
                if proj == "gate_up_proj":
                    param_name = f"{prefix_part}.w13"
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    weight = tensor.transpose(-1, -2)
                    E = weight.shape[0]
                    full_inter = weight.shape[1] // 2
                    tp = param.shape[1] // 2
                    rank = 0
                    if full_inter != tp:
                        from .tp import _tp_rank
                        rank = _tp_rank()
                    gate = weight[:, :full_inter, :]
                    up = weight[:, full_inter:, :]
                    param.data[:, :tp, :].copy_(gate[:, rank * tp:(rank + 1) * tp, :])
                    param.data[:, tp:, :].copy_(up[:, rank * tp:(rank + 1) * tp, :])
                else:
                    param_name = f"{prefix_part}.w2"
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    weight = tensor.transpose(-1, -2)
                    full_inter = weight.shape[2]
                    tp_inter = param.shape[2]
                    rank = 0
                    if full_inter != tp_inter:
                        from .tp import _tp_rank
                        rank = _tp_rank()
                    param.data.copy_(weight[:, :, rank * tp_inter:(rank + 1) * tp_inter])
                loaded += 1
                continue

        if is_qwen2_vl:
            m_merger = _QWEN2_MERGER_RE.match(mapped_name)
            if m_merger:
                prefix, attr, wb = m_merger.groups()
                remap = {"ln_q": "norm", "mlp.0": "fc1", "mlp.2": "fc2"}
                mapped_name = f"{prefix}.{remap[attr]}.{wb}"

        if is_qwen3_vl:
            m_mlp = _QWEN3_VISION_MLP_RE.match(mapped_name)
            if m_mlp:
                prefix, fc_num, wb = m_mlp.groups()
                mapped_name = f"{prefix}.fc{fc_num}.{wb}"

            m_merger = _QWEN3_MERGER_FC_RE.match(mapped_name)
            if m_merger and "merger" in mapped_name:
                prefix, fc_name, wb = m_merger.groups()
                fc = "fc1" if fc_name == "linear_fc1" else "fc2"
                mapped_name = f"{prefix}.{fc}.{wb}"

        if is_qwen3_vl:
            if _VISION_POS_EMBED_RE.match(mapped_name):
                mapped_name = "visual.pos_embed_interp.pos_embed"

        if is_qwen_vl:
            m = _VISION_PATCH_EMBED_RE.match(mapped_name)
            if m:
                prefix, wb = m.groups()
                mapped_name = f"{prefix}.conv.{wb}"
            m = _VISION_BLOCK_NORM_RE.match(mapped_name)
            if m:
                prefix, wb = m.groups()
                mapped_name = f"{prefix}.norm.{wb}"
            m = _VISION_MERGER_NORM_RE.match(mapped_name)
            if m:
                prefix, wb = m.groups()
                mapped_name = f"{prefix}.norm.{wb}"

        if is_qwen_vl:
            m_qkv = _VISION_QKV_RE.match(mapped_name)
            if m_qkv:
                prefix, wb = m_qkv.groups()
                loaded += _load_vision_qkv(model, prefix, tensor, wb)
                continue

        m_scale = _WEIGHT_SCALE_INV_RE.match(mapped_name)
        if m_scale:
            layer_prefix = m_scale.group(1)
            matched_scale = False
            for orig_key, (packed_name, shard_id) in packed.items():
                if orig_key in layer_prefix:
                    scale_param_name = layer_prefix.replace(
                        orig_key, packed_name) + ".weight_scale_inv"
                    try:
                        param = model.get_parameter(scale_param_name)
                    except AttributeError:
                        break
                    scale_loader = getattr(param, "weight_loader", None)
                    if scale_loader:
                        scale_loader(param, tensor, shard_id)
                    else:
                        default_weight_loader(param, tensor)
                    loaded += 1
                    matched_scale = True
                    break
            if matched_scale:
                continue
            scale_param_name = layer_prefix + ".weight_scale_inv"
            try:
                param = model.get_parameter(scale_param_name)
            except AttributeError:
                continue
            scale_loader = getattr(param, "weight_loader", default_weight_loader)
            scale_loader(param, tensor)
            loaded += 1
            continue

        m = _EXPERT_RE.match(mapped_name)
        if m:
            moe_prefix, expert_id_str, w_name = m.groups()
            expert_id = int(expert_id_str)
            if w_name in ("w1", "w3"):
                param_name = f"{moe_prefix}.w13"
                param = model.get_parameter(param_name)
                param.weight_loader(
                    param, tensor, expert_id, is_w1=(w_name == "w1"),
                )
            else:
                param_name = f"{moe_prefix}.w2"
                param = model.get_parameter(param_name)
                param.weight_loader(param, tensor, expert_id)
            loaded += 1
            continue

        m_ds = _DEEPSEEK_EXPERT_RE.match(mapped_name)
        if m_ds:
            moe_prefix, expert_id_str, proj_name, attr = m_ds.groups()
            expert_id = int(expert_id_str)
            if proj_name in ("gate_proj", "up_proj"):
                is_w1 = (proj_name == "gate_proj")
                if attr == "weight":
                    param_name = f"{moe_prefix}.w13"
                else:
                    param_name = f"{moe_prefix}.w13_weight_scale_inv"
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    continue
                param.weight_loader(param, tensor, expert_id, is_w1=is_w1)
            else:
                if attr == "weight":
                    param_name = f"{moe_prefix}.w2"
                else:
                    param_name = f"{moe_prefix}.w2_weight_scale_inv"
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    continue
                param.weight_loader(param, tensor, expert_id)
            loaded += 1
            continue

        matched = False
        for orig_key, (packed_name, shard_id) in packed.items():
            if is_llama4 and "experts." in mapped_name:
                continue
            if orig_key in mapped_name:
                param_name = mapped_name.replace(orig_key, packed_name)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    break
                weight_loader = getattr(param, "weight_loader")
                if is_llama4 and orig_key in ("q_proj", "k_proj"):
                    n_heads = (
                        llama4_config.num_key_value_heads
                        if orig_key == "k_proj"
                        else llama4_config.num_attention_heads
                    )
                    tensor = _permute_qk_for_rotary(tensor, n_heads)
                weight_loader(param, tensor, shard_id)
                loaded += 1
                matched = True
                break
        if matched:
            continue
        if "rotary_emb" in mapped_name:
            continue
        try:
            param = model.get_parameter(mapped_name)
        except AttributeError:
            continue
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, tensor)
        loaded += 1
    print(f"  Loaded {loaded} weight shards ({_time.perf_counter()-_t_load:.1f}s).")


def _postprocess_fp8_weights(model: torch.nn.Module) -> None:
    """Re-quantize FP8 weights to UE8M0 format and transform scale layout for DeepGEMM."""
    import time as _time
    _t_pp = _time.perf_counter()
    print("  Post-processing FP8 weights for DeepGEMM...", flush=True)
    linear_modules = [
        m for m in model.modules()
        if isinstance(getattr(m, 'linear_op', None), Fp8Linear)
    ]
    for i, module in enumerate(linear_modules):
        w = module.weight
        s = module.weight_scale_inv
        w_new, s_new = postprocess_fp8_weights(w.data, s.data)
        w.data.copy_(w_new)
        s.data.copy_(s_new)
    print(f"    {len(linear_modules)} FP8 linear layers done "
          f"({_time.perf_counter()-_t_pp:.1f}s)", flush=True)

    from ..tasks.baseline.L2.deepseek_moe import DeepSeekMoE
    moe_modules = [
        m for m in model.modules()
        if isinstance(m, DeepSeekMoE) and m.use_fp8
    ]
    moe_count = 0
    if moe_modules:
        total_ops = len(moe_modules) * 2
        _t_moe = _time.perf_counter()
        for j, module in enumerate(moe_modules):
            for name in ("w13", "w2"):
                w = getattr(module, name)
                s = getattr(module, f"{name}_weight_scale_inv")
                postprocess_fp8_weights_batched(w.data, s.data)
                moe_count += w.shape[0]
            done = (j + 1) * 2
            if j % max(1, len(moe_modules) // 5) == 0 or j == len(moe_modules) - 1:
                print(f"    MoE postprocess {done}/{total_ops} "
                      f"({done*100//total_ops}%, "
                      f"{_time.perf_counter()-_t_moe:.1f}s)", flush=True)
    print(f"  Post-processed {len(linear_modules)} FP8 linear layers, "
          f"{moe_count} MoE expert weight slices "
          f"({_time.perf_counter()-_t_pp:.1f}s total).", flush=True)


def _detect_model_type(model_name: str) -> str:
    """Detect model architecture from HuggingFace config."""
    try:
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(hf_config, "model_type", "llama")
    except ValueError:
        from huggingface_hub import hf_hub_download
        import json
        path = hf_hub_download(model_name, "config.json")
        with open(path) as f:
            model_type = json.load(f).get("model_type", "llama")
    if model_type == "deepseek_v32":
        model_type = "deepseek_v3"
    return model_type


def _detect_quant_config(model_name: str) -> dict | None:
    """Detect FP8 quantization config from HuggingFace config."""
    try:
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except ValueError:
        from huggingface_hub import hf_hub_download
        import json
        path = hf_hub_download(model_name, "config.json")
        with open(path) as f:
            cfg = json.load(f)
        from types import SimpleNamespace
        hf_config = SimpleNamespace(**cfg)
    qc = getattr(hf_config, "quantization_config", None)
    if qc is None:
        return None
    if isinstance(qc, dict):
        quant_method = qc.get("quant_method", "")
    else:
        quant_method = getattr(qc, "quant_method", "")
    if quant_method != "fp8":
        return None
    if isinstance(qc, dict):
        return qc
    return qc.to_dict() if hasattr(qc, "to_dict") else {"quant_method": "fp8"}


def load_model(
    model_name: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    model_path = download_model(model_name)
    model_type = _detect_model_type(model_name)
    quant_config = _detect_quant_config(model_name)

    if quant_config:
        print(f"  Detected FP8 quantization: {quant_config.get('quant_method')}, "
              f"block_size={quant_config.get('weight_block_size')}")

    if model_type == "llama4":
        config = Llama4Config.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Llama4 model ({config.num_local_experts} experts, "
              f"top-{config.num_experts_per_tok})...")
        model = Llama4ForCausalLM(config)
    elif model_type == "mixtral":
        config = MixtralConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Mixtral model ({config.num_local_experts} experts)...")
        model = MixtralForCausalLM(config)
    elif model_type == "qwen2_vl":
        config = Qwen2VLConfig.from_pretrained(model_name)
        config.dtype = dtype
        print("  Allocating Qwen2-VL model...")
        model = Qwen2VLForConditionalGeneration(config)
    elif model_type == "qwen3_vl":
        config = Qwen3VLConfig.from_pretrained(model_name)
        config.dtype = dtype
        print("  Allocating Qwen3-VL model...")
        model = Qwen3VLForConditionalGeneration(config, quant_config=quant_config)
    elif model_type == "deepseek_v3":
        config = DeepSeekV3Config.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating DeepSeek V3.2 model ({config.n_routed_experts} experts, "
              f"top-{config.num_experts_per_tok}, DSA topk={config.index_topk})...")
        model = DeepSeekV3ForCausalLM(config, quant_config=quant_config)
    else:
        config = LlamaConfig.from_pretrained(model_name)
        config.dtype = dtype
        print("  Allocating Llama model...")
        model = LlamaForCausalLM(config)

    tp = _tp_size()
    if hasattr(config, "num_attention_heads") and config.num_attention_heads % tp != 0:
        raise ValueError(
            f"TP degree {tp} is incompatible with {config.num_attention_heads} Q heads "
            f"(num_attention_heads must be divisible by tensor_parallel_size)"
        )
    if (model_type != "deepseek_v3"
            and hasattr(config, "num_key_value_heads")
            and config.num_key_value_heads % tp != 0):
        raise ValueError(
            f"TP degree {tp} is incompatible with {config.num_key_value_heads} KV heads "
            f"(num_key_value_heads must be divisible by tensor_parallel_size)"
        )

    load_weights(model, model_path, model_type)

    if quant_config:
        for name, param in model.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                if not param.is_cuda:
                    param.data = param.data.to(device=device)
            elif "weight_scale_inv" in name:
                param.data = param.data.to(device=device)
            elif param.data.device != device or param.dtype != dtype:
                param.data = param.data.to(device=device, dtype=dtype)
        for name, buf in model.named_buffers():
            if buf.device != device:
                buf.data = buf.data.to(device=device)
        if model_type == "deepseek_v3":
            _compute_mla_absorbed_weights(model)
        _postprocess_fp8_weights(model)
    else:
        model = model.to(device=device, dtype=dtype)

    if model_type == "deepseek_v3" and quant_config is None:
        _compute_mla_absorbed_weights(model)

    model.eval()
    return model, config


def _compute_mla_absorbed_weights(model: torch.nn.Module) -> None:
    """Compute absorbed W_UV weights for MLA decode after loading."""
    from ..tasks.baseline.L2.deepseek_mla_attention import DeepSeekMLAAttention
    count = 0
    for module in model.modules():
        if isinstance(module, DeepSeekMLAAttention):
            module.compute_absorbed_weights()
            count += 1
    print(f"  Computed absorbed MLA weights for {count} attention layers.")
