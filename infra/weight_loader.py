"""
Weight loader for Llama 3.1, Mixtral, Qwen2-VL, Qwen3-VL, and Qwen3-VL-MoE
with tensor parallelism.

Loads weights from HuggingFace safetensors and distributes them
across TP shards using the weight_loader callbacks on each parameter.
Supports FP8 quantized models (float8_e4m3fn weights + block scale factors).
"""

from __future__ import annotations

import gc
import os
import re
from glob import glob
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

from ..tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from ..tasks.baseline.L4.mixtral import MixtralConfig, MixtralForCausalLM
from ..tasks.baseline.L4.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from ..tasks.baseline.L4.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from ..tasks.baseline.L4.qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def download_model(model_name: str) -> str:
    return snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )


_EXPERT_RE = re.compile(
    r"(.+\.block_sparse_moe)\.experts\.(\d+)\.(w[123])\.weight"
)

# Qwen3-VL-MoE fused 3D expert tensors (no per-expert index in name)
_FUSED_EXPERT_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)$"
)
_FUSED_EXPERT_SCALE_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)_scale_inv$"
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


def _is_scale_inv_tensor(name: str) -> tuple[bool, str]:
    """Check if a tensor name is a weight_scale_inv and return the base weight name.

    FP8 checkpoints store scale factors as e.g.:
      model.layers.0.self_attn.q_proj.weight_scale_inv
    This maps to the weight_scale_inv parameter on the packed linear.
    """
    if name.endswith(".weight_scale_inv"):
        base = name[: -len(".weight_scale_inv")]
        return True, base
    return False, name


def load_weights(model, model_path: str, model_type: str = "llama") -> None:
    """Load weights with support for packed modules, MoE experts, vision
    encoder QKV, TP sharding, and FP8 scale_inv tensors.
    """
    packed = getattr(model, "packed_modules_mapping", {})
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    is_qwen2_vl = model_type == "qwen2_vl"
    is_qwen3_vl = model_type in ("qwen3_vl", "qwen3_vl_moe")
    is_qwen_vl = is_qwen2_vl or is_qwen3_vl

    print(f"  Loading weights from {len(safetensor_files)} safetensors file(s)...")
    loaded = 0
    for sf_file in safetensor_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Remap checkpoint names for Qwen VL models
                if is_qwen2_vl:
                    mapped_name = _remap_qwen2_vl_name(weight_name)
                elif is_qwen3_vl:
                    mapped_name = _remap_qwen3_vl_name(weight_name)
                else:
                    mapped_name = weight_name

                # Detect scale_inv tensors -- remap before further processing
                is_scale, scale_base = _is_scale_inv_tensor(mapped_name)
                if is_scale:
                    mapped_name = scale_base + ".weight_scale_inv"

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

                    # Handle Qwen3-VL merger: linear_fc1 -> fc1, linear_fc2 -> fc2
                    m_merger = _QWEN3_MERGER_FC_RE.match(mapped_name)
                    if m_merger and "merger" in mapped_name:
                        prefix, fc_name, wb = m_merger.groups()
                        fc = "fc1" if fc_name == "linear_fc1" else "fc2"
                        mapped_name = f"{prefix}.{fc}.{wb}"

                # Remap learned pos embed nesting (Qwen3-VL)
                if is_qwen3_vl:
                    if _VISION_POS_EMBED_RE.match(mapped_name):
                        mapped_name = "visual.pos_embed_interp.pos_embed"

                # Remap vision param names for L1 wrapper nesting
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

                # Handle vision encoder merged QKV weights
                if is_qwen_vl:
                    m_qkv = _VISION_QKV_RE.match(mapped_name)
                    if m_qkv:
                        prefix, wb = m_qkv.groups()
                        loaded += _load_vision_qkv(
                            model, prefix, f.get_tensor(weight_name), wb,
                        )
                        continue

                # Handle Qwen3-VL-MoE fused 3D expert weights + scales
                m_fused = _FUSED_EXPERT_RE.match(mapped_name)
                if m_fused:
                    mlp_prefix, proj_name = m_fused.groups()
                    if proj_name == "gate_up_proj":
                        param_name = f"{mlp_prefix}.w13"
                    else:
                        param_name = f"{mlp_prefix}.w2"
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    param.weight_loader(param, f.get_tensor(weight_name))
                    loaded += 1
                    continue

                m_fused_scale = _FUSED_EXPERT_SCALE_RE.match(mapped_name)
                if m_fused_scale:
                    mlp_prefix, proj_name = m_fused_scale.groups()
                    if proj_name == "gate_up_proj":
                        param_name = f"{mlp_prefix}.w13_scale_inv"
                    else:
                        param_name = f"{mlp_prefix}.w2_scale_inv"
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded += 1
                    continue

                # Handle per-expert MoE weights (Mixtral style)
                m = _EXPERT_RE.match(mapped_name)
                if m:
                    moe_prefix, expert_id_str, w_name = m.groups()
                    expert_id = int(expert_id_str)
                    if w_name in ("w1", "w3"):
                        param_name = f"{moe_prefix}.w13"
                        param = model.get_parameter(param_name)
                        param.weight_loader(
                            param, f.get_tensor(weight_name),
                            expert_id, is_w1=(w_name == "w1"),
                        )
                    else:
                        param_name = f"{moe_prefix}.w2"
                        param = model.get_parameter(param_name)
                        param.weight_loader(
                            param, f.get_tensor(weight_name), expert_id,
                        )
                    loaded += 1
                    continue

                # Handle packed modules (qkv_proj, gate_up_proj) and their scale_inv
                matched = False
                for orig_key, (packed_name, shard_id) in packed.items():
                    if is_scale:
                        scale_suffix = ".weight_scale_inv"
                        check_key = orig_key + scale_suffix
                        if check_key in mapped_name:
                            param_name = mapped_name.replace(
                                check_key, packed_name + scale_suffix
                            )
                            try:
                                param = model.get_parameter(param_name)
                            except AttributeError:
                                break
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            loaded += 1
                            matched = True
                            break
                    else:
                        if orig_key in mapped_name:
                            param_name = mapped_name.replace(orig_key, packed_name)
                            try:
                                param = model.get_parameter(param_name)
                            except AttributeError:
                                break
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
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
                weight_loader(param, f.get_tensor(weight_name))
                loaded += 1
    print(f"  Loaded {loaded} weight shards.")


def _detect_model_type(model_name: str) -> str:
    """Detect model architecture from HuggingFace config."""
    hf_config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(hf_config, "model_type", "llama")
    return model_type


def _move_model_to_device(model: torch.nn.Module, device: torch.device,
                          dtype: torch.dtype) -> torch.nn.Module:
    """Move model to device, skipping dtype cast for FP8 parameters."""
    has_fp8 = any(
        p.dtype == torch.float8_e4m3fn for p in model.parameters()
    )
    if not has_fp8:
        return model.to(device=device, dtype=dtype)

    model = model.to(device=device)
    for param in model.parameters():
        if param.dtype != torch.float8_e4m3fn:
            param.data = param.data.to(dtype=dtype)
    return model


def load_model(
    model_name: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    model_path = download_model(model_name)
    model_type = _detect_model_type(model_name)

    if model_type == "mixtral":
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
        model = Qwen3VLForConditionalGeneration(config)
    elif model_type == "qwen3_vl_moe":
        config = Qwen3VLMoeConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Qwen3-VL-MoE model ({config.num_experts} experts, "
              f"top-{config.num_experts_per_tok})...")
        model = Qwen3VLMoeForConditionalGeneration(config)
    else:
        config = LlamaConfig.from_pretrained(model_name)
        config.dtype = dtype
        print("  Allocating Llama model...")
        model = LlamaForCausalLM(config)

    load_weights(model, model_path, model_type)
    model = _move_model_to_device(model, device, dtype)
    model.eval()
    return model, config
