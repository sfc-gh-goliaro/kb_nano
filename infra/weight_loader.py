"""
Weight loader for Llama 3.1, Llama 4, Mixtral, Qwen2-VL, Qwen3-VL,
GPT-OSS, and Whisper with tensor parallelism.

Loads weights from HuggingFace safetensors and distributes them
across TP shards using the weight_loader callbacks on each parameter.

GPT-OSS uses a dedicated loader (_load_gpt_oss_weights) that keeps
expert weights in native MXFP4 packed uint8 format for Triton inference.
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

from concurrent.futures import ThreadPoolExecutor

from .tp import _tp_size
from ..tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from ..tasks.baseline.L4.llama_eagle3 import LlamaEagle3Config, LlamaForCausalLMEagle3
from ..tasks.baseline.L4.llama4 import Llama4Config, Llama4ForCausalLM
from ..tasks.baseline.L4.mixtral import MixtralConfig, MixtralForCausalLM
from ..tasks.baseline.L4.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from ..tasks.baseline.L4.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from ..tasks.baseline.L4.flux import FluxConfig, FluxPipeline
from ..tasks.baseline.L4.whisper import WhisperConfig, WhisperForConditionalGeneration
from ..tasks.baseline.L4.cosyvoice3 import CosyVoice3Config, CosyVoice3ForTTS


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def download_model(model_name: str) -> str:
    return snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )


_EXPERT_RE = re.compile(
    r"(.+\.block_sparse_moe)\.experts\.(\d+)\.(w[123])\.weight"
)

# Qwen3-MoE fused expert weight patterns: gate_up_proj [E, 2*inter, hidden], down_proj [E, hidden, inter]
_QWEN3_MOE_FUSED_EXPERT_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)$"
)
_QWEN3_MOE_FUSED_SCALE_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)_scale_inv$"
)

# Qwen3-MoE gate (router) weight
_QWEN3_MOE_GATE_RE = re.compile(r"(.+\.mlp)\.gate\.weight$")


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

# L1 wrapper nesting: *.embed_tokens.weight / *.lm_head.weight -> *.embedding_op.emb.weight
_EMBED_WEIGHT_RE = re.compile(
    r"((?:model\.)?(?:embed_tokens|lm_head))\.weight$"
)

# L1 wrapper nesting: patch_embed.proj.X -> patch_embed.proj.conv.X
_VISION_PATCH_EMBED_RE = re.compile(r"(visual\.patch_embed\.proj)\.(weight|bias)")


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


_FP4_E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def _dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
                     dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize MXFP4 weights to dense dtype (legacy fallback).

    For GPT-OSS, prefer keeping weights in MXFP4 format and using the
    native Triton MXFP4 MoE kernel. This function is only used by models
    that don't support native MXFP4 inference.
    """
    low = (blocks & 0x0F).long()
    high = ((blocks >> 4) & 0x0F).long()
    unpacked = torch.stack([low, high], dim=-1)
    unpacked = unpacked.reshape(*blocks.shape[:-1], 32)

    values = _FP4_E2M1_LUT[unpacked]

    scale_float = torch.pow(2.0, scales.float() - 127.0)

    values = values * scale_float.unsqueeze(-1)

    batch_shape = values.shape[:-2]
    values = values.reshape(*batch_shape, -1)
    return values.to(dtype)


_GPT_OSS_EXPERT_RE = re.compile(
    r"(model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)_(blocks|scales|bias)"
)


def _load_gpt_oss_weights(model, model_path: str) -> None:
    """Load GPT-OSS weights keeping expert weights in native MXFP4 format.

    Expert blocks/scales are loaded directly as packed uint8 tensors into
    the model's MXFP4 parameters. No dequantization is performed.
    """
    import gc

    packed = getattr(model, "packed_modules_mapping", {})
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    print(f"  Loading GPT-OSS weights from {len(safetensor_files)} safetensors file(s)...")

    # Map checkpoint names to model parameter names:
    #   model.layers.{i}.mlp.experts.gate_up_proj_blocks -> ...mlp.w13_weight
    #   model.layers.{i}.mlp.experts.gate_up_proj_scales -> ...mlp.w13_weight_scale
    #   model.layers.{i}.mlp.experts.gate_up_proj_bias   -> ...mlp.w13_bias
    #   model.layers.{i}.mlp.experts.down_proj_blocks    -> ...mlp.w2_weight
    #   model.layers.{i}.mlp.experts.down_proj_scales    -> ...mlp.w2_weight_scale
    #   model.layers.{i}.mlp.experts.down_proj_bias      -> ...mlp.w2_bias
    _EXPERT_MAP = {
        ("gate_up_proj", "blocks"): "w13_weight",
        ("gate_up_proj", "scales"): "w13_weight_scale",
        ("gate_up_proj", "bias"): "w13_bias",
        ("down_proj", "blocks"): "w2_weight",
        ("down_proj", "scales"): "w2_weight_scale",
        ("down_proj", "bias"): "w2_bias",
    }

    loaded = 0

    for sf_file in safetensor_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for weight_name in f.keys():
                m = _GPT_OSS_EXPERT_RE.match(weight_name)
                if m:
                    prefix, proj, part = m.groups()
                    param_suffix = _EXPERT_MAP.get((proj, part))
                    if param_suffix is None:
                        continue
                    param_name = f"{prefix.replace('mlp.experts', 'mlp')}.{param_suffix}"
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    tensor = f.get_tensor(weight_name)
                    param.weight_loader(param, tensor)
                    loaded += 1
                    continue

                # Non-expert weights
                tensor = f.get_tensor(weight_name)
                matched = False
                for orig_key, (packed_name, shard_id) in packed.items():
                    if orig_key in weight_name:
                        param_name = weight_name.replace(orig_key, packed_name)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            break
                        param.weight_loader(param, tensor, shard_id)
                        loaded += 1
                        matched = True
                        break
                if matched:
                    continue

                if "rotary_emb" in weight_name:
                    continue

                # Remap checkpoint names to model parameter names where
                # the module hierarchy differs (VocabParallelEmbedding
                # wraps the weight inside embedding_op.emb).
                mapped_name = weight_name
                _PARAM_REMAP = {
                    "model.embed_tokens.weight": "model.embed_tokens.embedding_op.emb.weight",
                    "lm_head.weight": "lm_head.embedding_op.emb.weight",
                }
                if weight_name in _PARAM_REMAP:
                    mapped_name = _PARAM_REMAP[weight_name]

                try:
                    param = model.get_parameter(mapped_name)
                except AttributeError:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
                loaded += 1

    gc.collect()
    print(f"  Loaded {loaded} weight shards.")


_WEIGHT_SCALE_INV_RE = re.compile(r"(.+)\.weight_scale_inv$")

# Whisper: remap checkpoint names
# Strip "model." prefix, fc1/fc2 -> mlp.fc1/mlp.fc2,
# conv -> conv.conv (L1 wrapper), layernorm -> layernorm.norm (L1 wrapper)
_WHISPER_FC_RE = re.compile(
    r"((?:encoder|decoder)\.layers\.\d+)\.fc([12])\.(weight|bias)"
)
_WHISPER_CONV_RE = re.compile(r"(encoder\.conv[12])\.(weight|bias)")
_WHISPER_LAYER_NORM_RE = re.compile(
    r"((?:encoder|decoder)(?:\.layers\.\d+)?\.(?:self_attn_layer_norm|"
    r"encoder_attn_layer_norm|final_layer_norm|layer_norm))\.(weight|bias)"
)
_WHISPER_EMBED_RE = re.compile(
    r"((?:encoder|decoder)\.embed_(?:positions|tokens))\.weight"
)
_WHISPER_OUT_PROJ_RE = re.compile(
    r"((?:encoder|decoder)\.layers\.\d+\.(?:self_attn|encoder_attn))\.out_proj\.(weight|bias)"
)


def _dequant_fp8_block(tensor: torch.Tensor, scale_inv: torch.Tensor,
                       block_size: int = 128) -> torch.Tensor:
    """Dequantize FP8 block-quantized tensor: out = fp8_val * scale_inv (per block).

    Each block of block_size elements along each non-batch dim shares one scale factor.
    Supports 2D [R, C] and 3D [E, R, C] tensors.
    """
    shape = tensor.shape
    ndim = len(shape)
    if ndim == 3:
        E, R, C = shape
        _, sR, sC = scale_inv.shape
        bR = (R + sR - 1) // sR
        bC = (C + sC - 1) // sC
        out = torch.zeros(E, sR * bR, sC * bC, dtype=torch.bfloat16, device=tensor.device)
        out[:, :R, :C] = tensor.to(torch.bfloat16)
        out = out.reshape(E, sR, bR, sC, bC) * scale_inv[:, :, None, :, None]
        return out.reshape(E, sR * bR, sC * bC)[:, :R, :C].contiguous()
    elif ndim == 2:
        R, C = shape
        sR, sC = scale_inv.shape
        bR = (R + sR - 1) // sR
        bC = (C + sC - 1) // sC
        out = torch.zeros(sR * bR, sC * bC, dtype=torch.bfloat16, device=tensor.device)
        out[:R, :C] = tensor.to(torch.bfloat16)
        out = out.reshape(sR, bR, sC, bC) * scale_inv[:, None, :, None]
        return out.reshape(sR * bR, sC * bC)[:R, :C].contiguous()
    else:
        raise ValueError(f"Unsupported tensor ndim={ndim} for FP8 dequantization")


def _threaded_safetensors_iterator(safetensor_files):
    """Yield (weight_name, tensor) with threaded pre-loading of safetensors files."""
    def _load_one(path):
        tensors = {}
        with safe_open(path, "pt", "cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors

    with ThreadPoolExecutor(max_workers=min(4, len(safetensor_files))) as pool:
        futures = [pool.submit(_load_one, sf) for sf in sorted(safetensor_files)]
        for fut in futures:
            tensors = fut.result()
            for k, v in tensors.items():
                yield k, v
            del tensors


def _fastsafetensors_iterator(safetensor_files):
    """Yield (weight_name, tensor) using fastsafetensors GPU-direct loading."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    pg = SingleGroup()
    sorted_files = sorted(safetensor_files)

    for f_path in sorted_files:
        loader = SafeTensorsFileLoader(pg, device, nogds=True)
        loader.add_filenames({0: [f_path]})
        try:
            fb = loader.copy_files_to_device()
            try:
                for k in list(fb.key_to_rank_lidx.keys()):
                    yield k, fb.get_tensor(k)
            finally:
                fb.close()
        finally:
            loader.close()


def _assign_fused_expert(model, key, tensor, scale):
    """Assign a single fused expert weight+scale to the model, then free them.

    Handles FP8 TP-sharded assignment with scale transposition.
    Called as soon as both weight and scale are available for a given
    (mlp_prefix, proj) key, avoiding buffering all layers simultaneously.
    """
    from .tp import _tp_rank
    mlp_prefix, proj = key
    rank = _tp_rank() if _tp_size() > 1 else 0
    is_fp8 = tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if proj == "gate_up_proj":
        param_name = f"{mlp_prefix}.w13"
    else:
        param_name = f"{mlp_prefix}.w2"
    try:
        param = model.get_parameter(param_name)
    except AttributeError:
        return

    model_is_fp8 = param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if is_fp8 and model_is_fp8 and scale is not None:
        weight = tensor.transpose(-1, -2)
        if proj == "gate_up_proj":
            full_inter_2 = weight.shape[1]
            half = full_inter_2 // 2
            tp = param.shape[1] // 2
            gate = weight[:, :half, :]
            up = weight[:, half:, :]
            param.data[:, :tp, :].copy_(gate[:, rank * tp:(rank + 1) * tp, :])
            param.data[:, tp:, :].copy_(up[:, rank * tp:(rank + 1) * tp, :])

            scale_param_name = f"{mlp_prefix}.w13_scale"
            try:
                scale_param = model.get_parameter(scale_param_name)
            except AttributeError:
                pass
            else:
                s = scale.transpose(1, 2)
                full_scale_rows = s.shape[1]
                gate_scale_rows = full_scale_rows // 2
                shard_scale_rows = scale_param.data.shape[1] // 2
                gate_s = s[:, rank * shard_scale_rows:(rank + 1) * shard_scale_rows, :]
                up_s = s[:, gate_scale_rows + rank * shard_scale_rows:gate_scale_rows + (rank + 1) * shard_scale_rows, :]
                scale_param.data[:, :shard_scale_rows, :].copy_(gate_s)
                scale_param.data[:, shard_scale_rows:2 * shard_scale_rows, :].copy_(up_s)
        else:
            full_inter = weight.shape[2]
            tp_inter = param.shape[2]
            param.data.copy_(weight[:, :, rank * tp_inter:(rank + 1) * tp_inter])

            scale_param_name = f"{mlp_prefix}.w2_scale"
            try:
                scale_param = model.get_parameter(scale_param_name)
            except AttributeError:
                pass
            else:
                s = scale.transpose(1, 2)
                shard_scale_cols = scale_param.data.shape[2]
                scale_param.data.copy_(s[:, :, rank * shard_scale_cols:(rank + 1) * shard_scale_cols])
    else:
        if is_fp8 and scale is not None:
            tensor = _dequant_fp8_block(tensor, scale, block_size=128)
        weight = tensor.transpose(-1, -2)
        if proj == "gate_up_proj":
            full_inter_2 = weight.shape[1]
            half = full_inter_2 // 2
            tp = param.shape[1] // 2
            gate = weight[:, :half, :]
            up = weight[:, half:, :]
            param.data[:, :tp, :].copy_(gate[:, rank * tp:(rank + 1) * tp, :])
            param.data[:, tp:, :].copy_(up[:, rank * tp:(rank + 1) * tp, :])
        else:
            full_inter = weight.shape[2]
            tp_inter = param.shape[2]
            param.data.copy_(weight[:, :, rank * tp_inter:(rank + 1) * tp_inter])


def load_weights(model, model_path: str, model_type: str = "llama") -> None:
    """Load weights with support for packed modules, MoE experts, vision
    encoder QKV, FP8 weight_scale_inv, and TP sharding.
    """
    packed = getattr(model, "packed_modules_mapping", {})
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    is_whisper = model_type == "whisper"
    is_qwen2_vl = model_type == "qwen2_vl"
    is_qwen3_vl = model_type in ("qwen3_vl", "qwen3_vl_moe")
    is_qwen3_vl_moe = model_type == "qwen3_vl_moe"
    is_qwen_vl = is_qwen2_vl or is_qwen3_vl
    is_llama4 = model_type == "llama4"
    if is_llama4:
        llama4_config = model.config

    if _HAS_FASTSAFETENSORS:
        print(f"  Loading weights from {len(safetensor_files)} safetensors file(s) "
              f"[fastsafetensors GPU-direct]...")
    elif len(safetensor_files) > 1:
        print(f"  Loading weights from {len(safetensor_files)} safetensors file(s) "
              f"[threaded]...")
    else:
        print(f"  Loading weights from {len(safetensor_files)} safetensors file(s)...")
    loaded = 0
    _fused_expert_weights = {}
    _fused_expert_scales = {}

    if _HAS_FASTSAFETENSORS:
        _weight_iter = _fastsafetensors_iterator(safetensor_files)
    elif len(safetensor_files) > 1:
        _weight_iter = _threaded_safetensors_iterator(safetensor_files)
    else:
        def _std_iter():
            for sf_file in safetensor_files:
                with safe_open(sf_file, "pt", "cpu") as f:
                    for wn in f.keys():
                        yield wn, f.get_tensor(wn)
        _weight_iter = _std_iter()

    for weight_name, _loaded_tensor in _weight_iter:
        def _get_tensor(_t=_loaded_tensor):
            return _t
        # Remap checkpoint names for Qwen VL models
        if is_qwen2_vl:
            mapped_name = _remap_qwen2_vl_name(weight_name)
        elif is_qwen3_vl:
            mapped_name = _remap_qwen3_vl_name(weight_name)
        else:
            mapped_name = weight_name

        # Whisper: remap checkpoint names
        if is_whisper:
            if mapped_name.startswith("model."):
                mapped_name = mapped_name[len("model."):]
            if mapped_name.startswith("proj_out."):
                continue
            m_fc = _WHISPER_FC_RE.match(mapped_name)
            if m_fc:
                prefix, fc_num, wb = m_fc.groups()
                mapped_name = f"{prefix}.mlp.fc{fc_num}.{wb}"
            m_conv = _WHISPER_CONV_RE.match(mapped_name)
            if m_conv:
                prefix, wb = m_conv.groups()
                mapped_name = f"{prefix}.conv.{wb}"
            m_ln = _WHISPER_LAYER_NORM_RE.match(mapped_name)
            if m_ln:
                prefix, wb = m_ln.groups()
                mapped_name = f"{prefix}.norm.{wb}"
            m_emb = _WHISPER_EMBED_RE.match(mapped_name)
            if m_emb:
                prefix = m_emb.group(1)
                mapped_name = f"{prefix}.emb.weight"
            if mapped_name.endswith(".k_proj.weight"):
                tensor = _get_tensor()
                fake_bias_name = mapped_name.replace(".weight", ".bias")
                fake_bias = torch.zeros(tensor.size(0))
                for orig_key, (packed_name, shard_id) in packed.items():
                    if orig_key in fake_bias_name:
                        param_name = fake_bias_name.replace(orig_key, packed_name)
                        try:
                            param = model.get_parameter(param_name)
                            weight_loader_fn = getattr(param, "weight_loader")
                            weight_loader_fn(param, fake_bias, shard_id)
                        except AttributeError:
                            pass
                        break

        # Llama4: strip language_model. prefix, skip vision weights
        if is_llama4:
            if not mapped_name.startswith("language_model."):
                continue
            mapped_name = mapped_name[len("language_model."):]
            m_fused = _LLAMA4_FUSED_EXPERT_RE.match(mapped_name)
            if m_fused:
                prefix_part, proj = m_fused.groups()
                tensor = _get_tensor()
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

        # Handle Qwen2-VL merger: ln_q -> norm, mlp.0 -> fc1, mlp.2 -> fc2
        if is_qwen2_vl:
            m_merger = _QWEN2_MERGER_RE.match(mapped_name)
            if m_merger:
                prefix, attr, wb = m_merger.groups()
                remap = {"ln_q": "norm", "mlp.0": "fc1", "mlp.2": "fc2"}
                mapped_name = f"{prefix}.{remap[attr]}.{wb}"

        # Handle Qwen3-VL vision MLP: linear_fc1 -> fc1, linear_fc2 -> fc2
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

        # Remap learned pos embed nesting (Qwen3-VL)
        if is_qwen3_vl:
            if _VISION_POS_EMBED_RE.match(mapped_name):
                mapped_name = "visual.pos_embed_interp._embed.emb.weight"

        # Remap vision param names for L1 wrapper nesting
        if is_qwen_vl:
            m = _VISION_PATCH_EMBED_RE.match(mapped_name)
            if m:
                prefix, wb = m.groups()
                mapped_name = f"{prefix}.conv.{wb}"

        # Handle vision encoder merged QKV weights
        if is_qwen_vl:
            m_qkv = _VISION_QKV_RE.match(mapped_name)
            if m_qkv:
                prefix, wb = m_qkv.groups()
                loaded += _load_vision_qkv(model, prefix, _get_tensor(), wb)
                continue

        # Handle FP8 weight_scale_inv tensors
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
                        scale_loader(param, _get_tensor(), shard_id)
                    else:
                        default_weight_loader(param, _get_tensor())
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
            scale_loader(param, _get_tensor())
            loaded += 1
            continue

        # Handle MoE expert weights
        m = _EXPERT_RE.match(mapped_name)
        if m:
            moe_prefix, expert_id_str, w_name = m.groups()
            expert_id = int(expert_id_str)
            if w_name in ("w1", "w3"):
                param_name = f"{moe_prefix}.w13"
                param = model.get_parameter(param_name)
                param.weight_loader(param, _get_tensor(), expert_id, is_w1=(w_name == "w1"))
            else:
                param_name = f"{moe_prefix}.w2"
                param = model.get_parameter(param_name)
                param.weight_loader(param, _get_tensor(), expert_id)
            loaded += 1
            continue

        # Handle Qwen3-VL-MoE fused 3D expert weights, scales, and gate
        if is_qwen3_vl_moe:
            m_gate = _QWEN3_MOE_GATE_RE.match(mapped_name)
            if m_gate:
                mlp_prefix = m_gate.group(1)
                param_name = f"{mlp_prefix}.gate.weight"
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    pass
                else:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, _get_tensor())
                    loaded += 1
                continue

            m_fused_scale = _QWEN3_MOE_FUSED_SCALE_RE.match(mapped_name)
            if m_fused_scale:
                mlp_prefix, proj = m_fused_scale.groups()
                _fused_expert_scales[(mlp_prefix, proj)] = _get_tensor()
                key = (mlp_prefix, proj)
                if key in _fused_expert_weights:
                    _assign_fused_expert(model, key, _fused_expert_weights.pop(key),
                                         _fused_expert_scales.pop(key))
                    loaded += 1
                continue

            m_fused = _QWEN3_MOE_FUSED_EXPERT_RE.match(mapped_name)
            if m_fused:
                mlp_prefix, proj = m_fused.groups()
                key = (mlp_prefix, proj)
                _fused_expert_weights[key] = _get_tensor()
                if key in _fused_expert_scales:
                    _assign_fused_expert(model, key, _fused_expert_weights.pop(key),
                                         _fused_expert_scales.pop(key))
                    loaded += 1
                else:
                    loaded += 1
                continue

        # Handle packed modules (qkv_proj, gate_up_proj)
        matched = False
        for orig_key, (packed_name, shard_id) in packed.items():
            if (is_llama4 or is_qwen3_vl_moe) and "experts." in mapped_name:
                continue
            if orig_key in mapped_name:
                param_name = mapped_name.replace(orig_key, packed_name)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    break
                weight_loader = getattr(param, "weight_loader")
                if is_llama4 and orig_key in ("q_proj", "k_proj"):
                    tensor = _get_tensor()
                    n_heads = (
                        llama4_config.num_key_value_heads
                        if orig_key == "k_proj"
                        else llama4_config.num_attention_heads
                    )
                    tensor = _permute_qk_for_rotary(tensor, n_heads)
                    weight_loader(param, tensor, shard_id)
                else:
                    weight_loader(param, _get_tensor(), shard_id)
                loaded += 1
                matched = True
                break
        if matched:
            continue
        if "rotary_emb" in mapped_name:
            continue
        m_emb_w = _EMBED_WEIGHT_RE.match(mapped_name)
        if m_emb_w:
            mapped_name = f"{m_emb_w.group(1)}.embedding_op.emb.weight"
        try:
            param = model.get_parameter(mapped_name)
        except AttributeError:
            continue
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, _get_tensor())
        loaded += 1

        # Whisper: duplicate decoder embed_tokens -> lm_head (tied weights)
        if is_whisper and mapped_name == "decoder.embed_tokens.emb.weight":
            lm_param = model.get_parameter("lm_head.embedding_op.emb.weight")
            lm_loader = getattr(lm_param, "weight_loader", default_weight_loader)
            lm_loader(lm_param, _get_tensor())
            loaded += 1

    # Assign any remaining buffered fused expert weights (weight arrived
    # in a different safetensors file than its scale, so the pair wasn't
    # complete during the main loop).
    if _fused_expert_weights:
        for key in list(_fused_expert_weights.keys()):
            scale = _fused_expert_scales.pop(key, None)
            _assign_fused_expert(model, key, _fused_expert_weights.pop(key), scale)
        del _fused_expert_weights, _fused_expert_scales

    print(f"  Loaded {loaded} weight shards.")


def _postprocess_moe_fp8_weights(module) -> int:
    """Post-process MoE expert FP8 weights for DeepGEMM scale layout.

    Keeps original FP8 weights and scales from the checkpoint (no UE8M0
    requantization) and creates DeepGEMM-layout transformed scale tensors
    stored in w13_scale_dg / w2_scale_dg.

    Only runs when DeepGEMM is available on Hopper+ GPUs.
    """
    import deep_gemm

    if not hasattr(module, 'w13') or not hasattr(module, 'w13_scale'):
        return 0
    if module.w13.dtype != torch.float8_e4m3fn:
        return 0

    from ..tasks.baseline.L1.moe_grouped_gemm import _is_deep_gemm_supported
    if not _is_deep_gemm_supported():
        return 0

    block_shape = getattr(module, 'block_shape', [128, 128])
    block_m, block_k = int(block_shape[0]), int(block_shape[1])

    count = 0
    for wname, sname in [('w13', 'w13_scale'), ('w2', 'w2_scale')]:
        wq = getattr(module, wname).data
        ws = getattr(module, sname).data

        E = wq.size(0)

        recipe = (1, block_m, block_k)
        dg_ws = deep_gemm.transform_sf_into_required_layout(
            sf=ws,
            mn=wq.size(1),
            k=wq.size(2),
            recipe=recipe,
            num_groups=E,
            is_sfa=False,
            disable_ue8m0_cast=True,
        )
        dg_sname = sname + "_dg"
        module.register_parameter(
            dg_sname, torch.nn.Parameter(dg_ws, requires_grad=False)
        )
        count += 1

    return count


def _postprocess_fp8_weights(model: torch.nn.Module) -> None:
    """Re-quantize FP8 weights to UE8M0 format and transform scale layout for DeepGEMM."""
    from ..tasks.baseline.L1.fp8_linear import Fp8Linear, postprocess_fp8_weights
    from ..tasks.baseline.L2.qwen3_moe import Qwen3MoE

    print("  Post-processing FP8 weights for DeepGEMM...")
    count = 0
    moe_count = 0
    for module in model.modules():
        if isinstance(module.linear_op if hasattr(module, 'linear_op') else None, Fp8Linear):
            w = module.weight
            s = module.weight_scale_inv
            w_new, s_new = postprocess_fp8_weights(w.data, s.data)
            module.weight = torch.nn.Parameter(w_new, requires_grad=False)
            module.weight_scale_inv = torch.nn.Parameter(s_new, requires_grad=False)
            count += 1
        elif isinstance(module, Qwen3MoE) and getattr(module, 'use_fp8', False):
            moe_count += _postprocess_moe_fp8_weights(module)
    if count > 0 or moe_count > 0:
        torch.cuda.empty_cache()
    print(f"  Post-processed {count} FP8 linear layers, {moe_count} MoE weight sets.")


def _is_diffusion_model(model_name: str) -> bool:
    """Check if the model is a diffusion model (e.g., FLUX, HunyuanVideo) by looking for model_index.json."""
    import json as _json
    model_path = model_name if os.path.isdir(model_name) else download_model(model_name)
    index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = _json.load(f)
        class_name = data.get("_class_name", "")
        if "Flux" in class_name or "Diffusion" in class_name or "HunyuanVideo" in class_name:
            return True
    return False


def _detect_diffusion_type(model_name: str) -> str:
    """Distinguish between diffusion model types (flux vs hunyuan_video)."""
    import json as _json
    model_path = model_name if os.path.isdir(model_name) else download_model(model_name)
    index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = _json.load(f)
        class_name = data.get("_class_name", "")
        if "HunyuanVideo" in class_name:
            return "hunyuan_video"
    return "flux"


def _detect_model_type(model_name: str) -> str:
    """Detect model architecture from HuggingFace config."""
    if _is_diffusion_model(model_name):
        return _detect_diffusion_type(model_name)
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(hf_config, "model_type", "llama")
    return model_type


def _detect_quant_config(model_name: str) -> dict | None:
    """Detect FP8 quantization config from HuggingFace config."""
    hf_config = AutoConfig.from_pretrained(model_name)
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

    if model_type in ("flux", "hunyuan_video"):
        raise ValueError(
            "Diffusion models should be loaded via "
            "kb_nano.infra.diffusion_engine.DiffusionEngine, "
            "not the LLM load_model() path."
        )
    if model_type == "cosyvoice3":
        raise ValueError(
            "CosyVoice3 TTS models should be loaded via "
            "kb_nano.infra.tts_engine.TTSEngine, "
            "not the LLM load_model() path."
        )
    if model_type == "gpt_oss":
        from ..tasks.baseline.L4.gpt_oss import GptOssConfig, GptOssForCausalLM
        config = GptOssConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating GPT-OSS model ({config.num_local_experts} experts, "
              f"top-{config.num_experts_per_tok})...")
        model = GptOssForCausalLM(config)
    elif model_type == "whisper":
        config = WhisperConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Whisper model (enc={config.encoder_layers}L, "
              f"dec={config.decoder_layers}L, d={config.d_model})...")
        model = WhisperForConditionalGeneration(config)
    elif model_type == "llama4":
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
    elif model_type in ("qwen3_vl", "qwen3_vl_moe"):
        config = Qwen3VLConfig.from_pretrained(model_name)
        config.dtype = dtype
        if config.is_moe:
            print(f"  Allocating Qwen3-VL-MoE model ({config.num_experts} experts, "
                  f"top-{config.num_experts_per_tok})...")
        else:
            print("  Allocating Qwen3-VL model...")
        model = Qwen3VLForConditionalGeneration(config, quant_config=quant_config)
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
    if hasattr(config, "num_key_value_heads") and config.num_key_value_heads % tp != 0:
        raise ValueError(
            f"TP degree {tp} is incompatible with {config.num_key_value_heads} KV heads "
            f"(num_key_value_heads must be divisible by tensor_parallel_size)"
        )

    if model_type == "gpt_oss":
        _load_gpt_oss_weights(model, model_path)
        # GPT-OSS has mixed dtypes: uint8 MXFP4 expert weights must not be cast
        for name, param in model.named_parameters():
            if param.dtype == torch.uint8:
                if param.data.device != device:
                    param.data = param.data.to(device=device)
            elif param.data.device != device or param.dtype != dtype:
                param.data = param.data.to(device=device, dtype=dtype)
        for name, buf in model.named_buffers():
            if buf.device != device:
                buf.data = buf.data.to(device=device)
        # Swizzle MXFP4 weights for Triton kernels after GPU transfer
        from ..tasks.baseline.L2.gpt_oss_moe import GptOssMoE
        for mod in model.modules():
            if isinstance(mod, GptOssMoE):
                mod.process_weights_after_loading()
    else:
        load_weights(model, model_path, model_type)

    if quant_config:
        for name, param in model.named_parameters():
            if param.dtype == torch.float8_e4m3fn:
                if not param.is_cuda:
                    param.data = param.data.to(device=device)
            elif "weight_scale_inv" in name or "w13_scale" in name or "w2_scale" in name:
                param.data = param.data.to(device=device)
            elif param.data.device != device or param.dtype != dtype:
                param.data = param.data.to(device=device, dtype=dtype)
        for name, buf in model.named_buffers():
            if buf.device != device:
                buf.data = buf.data.to(device=device)
        _postprocess_fp8_weights(model)
    elif model_type != "gpt_oss":
        model = model.to(device=device, dtype=dtype)

    model.eval()
    return model, config


def _iter_draft_weights(model_path: str):
    """Yield ``(name, tensor)`` from either safetensors or pytorch_model.bin.

    EAGLE-3 draft checkpoints are typically published as a single
    ``pytorch_model.bin`` file (HF legacy format). We also support
    safetensors for forward compatibility.
    """
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if safetensor_files:
        for sf_file in safetensor_files:
            with safe_open(sf_file, "pt", "cpu") as f:
                for k in f.keys():
                    yield k, f.get_tensor(k)
        return

    bin_files = sorted(glob(os.path.join(model_path, "pytorch_model*.bin")))
    if not bin_files:
        raise FileNotFoundError(
            f"No .safetensors or pytorch_model*.bin files found in {model_path}"
        )
    for bf in bin_files:
        state = torch.load(bf, map_location="cpu", weights_only=True)
        for k, v in state.items():
            yield k, v


def _load_eagle3_weights(model: LlamaForCausalLMEagle3, model_path: str) -> None:
    """Load weights for the EAGLE-3 draft model.

    Supports the published sglang draft checkpoints (e.g.
    ``jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B``). Handles the same
    packed_modules_mapping as the regular Llama loader, plus:
      - any ``*.d2t`` tensor is converted to ``hot_token_id = d2t + arange``
      - any ``*.t2d`` tensor is dropped
      - ``model.embed_tokens.weight`` and ``lm_head.weight`` are routed to the
        wrapped ``embedding_op.emb.weight`` parameter.
    """
    packed = getattr(model, "packed_modules_mapping", {})

    loaded = 0
    for weight_name, tensor in _iter_draft_weights(model_path):
        if "d2t" in weight_name:
            d2t = tensor.long()
            hot = d2t + torch.arange(d2t.shape[0], dtype=torch.long, device=d2t.device)
            model.hot_token_id.data.copy_(hot)
            model._has_hot_token_id = True
            loaded += 1
            continue

        if "t2d" in weight_name:
            continue

        mapped_name = weight_name
        if not mapped_name.startswith("model.") and not mapped_name.startswith("lm_head"):
            mapped_name = f"model.{mapped_name}"

        matched = False
        for orig_key, (packed_name, shard_id) in packed.items():
            if orig_key in mapped_name:
                param_name = mapped_name.replace(orig_key, packed_name)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    break
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, tensor, shard_id)
                loaded += 1
                matched = True
                break
        if matched:
            continue

        m_emb_w = _EMBED_WEIGHT_RE.match(mapped_name)
        if m_emb_w:
            mapped_name = f"{m_emb_w.group(1)}.embedding_op.emb.weight"

        try:
            param = model.get_parameter(mapped_name)
        except AttributeError:
            continue
        wl = getattr(param, "weight_loader", default_weight_loader)
        wl(param, tensor)
        loaded += 1

    print(f"  Loaded {loaded} EAGLE-3 weight shards.")


def load_eagle3_draft_model(
    draft_repo: str,
    target_config: LlamaConfig,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[LlamaForCausalLMEagle3, LlamaEagle3Config]:
    """Download and load an EAGLE-3 draft checkpoint.

    Supports both safetensors and pytorch_model.bin checkpoints. The returned
    model's ``embed_tokens`` is initialized from the checkpoint when present;
    the engine should still call
    ``model.set_embed_tokens(target.model.embed_tokens)`` to share memory with
    the target.
    """
    model_path = snapshot_download(
        draft_repo,
        allow_patterns=["*.safetensors", "*.json", "*.bin"],
    )
    config = LlamaEagle3Config.from_pretrained(draft_repo, target_config)
    config.dtype = dtype
    print(f"  Allocating EAGLE-3 draft model (hidden={config.hidden_size}, "
          f"draft_vocab={config.draft_vocab_size})...")
    model = LlamaForCausalLMEagle3(config)

    _load_eagle3_weights(model, model_path)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config
