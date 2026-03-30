"""
Weight loader for Llama 3.1, Llama 4, Mixtral, Qwen2-VL, Qwen3-VL,
and Whisper with tensor parallelism.

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

from .tp import _tp_size
from ..tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM
from ..tasks.baseline.L4.llama4 import Llama4Config, Llama4ForCausalLM
from ..tasks.baseline.L4.mixtral import MixtralConfig, MixtralForCausalLM
from ..tasks.baseline.L4.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from ..tasks.baseline.L4.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from ..tasks.baseline.L4.flux import FluxConfig, FluxPipeline
from ..tasks.baseline.L4.whisper import WhisperConfig, WhisperForConditionalGeneration


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def download_model(model_name: str) -> str:
    return snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )


_EXPERT_RE = re.compile(
    r"(.+\.block_sparse_moe)\.experts\.(\d+)\.(w[123])\.weight"
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

# L1 wrapper nesting: *.embed_tokens.weight / *.lm_head.weight -> *.embedding_op.emb.weight
_EMBED_WEIGHT_RE = re.compile(
    r"((?:model\.)?(?:embed_tokens|lm_head))\.weight$"
)

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
    is_qwen3_vl = model_type == "qwen3_vl"
    is_qwen_vl = is_qwen2_vl or is_qwen3_vl
    is_llama4 = model_type == "llama4"
    if is_llama4:
        llama4_config = model.config

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

                # Whisper: remap checkpoint names
                if is_whisper:
                    # Strip "model." prefix
                    if mapped_name.startswith("model."):
                        mapped_name = mapped_name[len("model."):]
                    # proj_out -> lm_head (tied weights, skip)
                    if mapped_name.startswith("proj_out."):
                        continue

                    # fc1/fc2 -> mlp.fc1/mlp.fc2
                    m_fc = _WHISPER_FC_RE.match(mapped_name)
                    if m_fc:
                        prefix, fc_num, wb = m_fc.groups()
                        mapped_name = f"{prefix}.mlp.fc{fc_num}.{wb}"

                    # conv1/conv2 -> conv1.conv/conv2.conv (L1 wrapper)
                    m_conv = _WHISPER_CONV_RE.match(mapped_name)
                    if m_conv:
                        prefix, wb = m_conv.groups()
                        mapped_name = f"{prefix}.conv.{wb}"

                    # LayerNorm -> *.norm.* (L1 wrapper)
                    m_ln = _WHISPER_LAYER_NORM_RE.match(mapped_name)
                    if m_ln:
                        prefix, wb = m_ln.groups()
                        mapped_name = f"{prefix}.norm.{wb}"

                    # Embedding L1 wrapper nesting: *.weight -> *.emb.weight
                    m_emb = _WHISPER_EMBED_RE.match(mapped_name)
                    if m_emb:
                        prefix = m_emb.group(1)
                        mapped_name = f"{prefix}.emb.weight"

                    # Create fake zero bias for k_proj (HF checkpoint has
                    # no k_proj bias, but our QKVParallelLinear expects one
                    # for the packed QKV)
                    if mapped_name.endswith(".k_proj.weight"):
                        tensor = f.get_tensor(weight_name)
                        fake_bias_name = mapped_name.replace(".weight", ".bias")
                        fake_bias = torch.zeros(tensor.size(0))
                        # Load the fake bias through packed mapping
                        for orig_key, (packed_name, shard_id) in packed.items():
                            if orig_key in fake_bias_name:
                                param_name = fake_bias_name.replace(
                                    orig_key, packed_name)
                                try:
                                    param = model.get_parameter(param_name)
                                    weight_loader_fn = getattr(
                                        param, "weight_loader")
                                    weight_loader_fn(param, fake_bias, shard_id)
                                except AttributeError:
                                    pass
                                break

                # Llama4: strip language_model. prefix, skip vision weights
                if is_llama4:
                    if not mapped_name.startswith("language_model."):
                        continue  # skip vision weights
                    mapped_name = mapped_name[len("language_model."):]

                    # Handle fused expert weights
                    m_fused = _LLAMA4_FUSED_EXPERT_RE.match(mapped_name)
                    if m_fused:
                        prefix_part, proj = m_fused.groups()
                        tensor = f.get_tensor(weight_name)
                        if proj == "gate_up_proj":
                            param_name = f"{prefix_part}.w13"
                            try:
                                param = model.get_parameter(param_name)
                            except AttributeError:
                                continue
                            # [E, hidden, 2*inter] → transpose → [E, 2*inter, hidden]
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
                        else:  # down_proj
                            param_name = f"{prefix_part}.w2"
                            try:
                                param = model.get_parameter(param_name)
                            except AttributeError:
                                continue
                            # [E, inter, hidden] → transpose → [E, hidden, inter]
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

                    # Handle Qwen3-VL merger: linear_fc1 -> fc1, linear_fc2 -> fc2
                    m_merger = _QWEN3_MERGER_FC_RE.match(mapped_name)
                    if m_merger and "merger" in mapped_name:
                        prefix, fc_name, wb = m_merger.groups()
                        fc = "fc1" if fc_name == "linear_fc1" else "fc2"
                        mapped_name = f"{prefix}.{fc}.{wb}"

                # Remap learned pos embed nesting (Qwen3-VL)
                if is_qwen3_vl:
                    if _VISION_POS_EMBED_RE.match(mapped_name):
                        mapped_name = "visual.pos_embed_interp._embed.weight"

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
                                scale_loader(param, f.get_tensor(weight_name), shard_id)
                            else:
                                default_weight_loader(param, f.get_tensor(weight_name))
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
                    scale_loader(param, f.get_tensor(weight_name))
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

                # Handle packed modules (qkv_proj, gate_up_proj)
                matched = False
                for orig_key, (packed_name, shard_id) in packed.items():
                    # Llama4: skip packed mapping for expert weight names
                    # (e.g. shared_expert.gate_proj should match, but
                    #  experts.gate_up_proj should not)
                    if is_llama4 and "experts." in mapped_name:
                        continue
                    if orig_key in mapped_name:
                        param_name = mapped_name.replace(orig_key, packed_name)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            break
                        weight_loader = getattr(param, "weight_loader")
                        # Llama4: use permuted Q/K weights for rotary
                        if is_llama4 and orig_key in ("q_proj", "k_proj"):
                            tensor = f.get_tensor(weight_name)
                            n_heads = (
                                llama4_config.num_key_value_heads
                                if orig_key == "k_proj"
                                else llama4_config.num_attention_heads
                            )
                            tensor = _permute_qk_for_rotary(tensor, n_heads)
                            weight_loader(param, tensor, shard_id)
                        else:
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
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
                weight_loader(param, f.get_tensor(weight_name))
                loaded += 1

                # Whisper: duplicate decoder embed_tokens -> lm_head (tied weights)
                if is_whisper and mapped_name == "decoder.embed_tokens.emb.weight":
                    lm_param = model.get_parameter("lm_head.embedding_op.emb.weight")
                    lm_loader = getattr(lm_param, "weight_loader", default_weight_loader)
                    lm_loader(lm_param, f.get_tensor(weight_name))
                    loaded += 1

    print(f"  Loaded {loaded} weight shards.")


def _postprocess_fp8_weights(model: torch.nn.Module) -> None:
    """Re-quantize FP8 weights to UE8M0 format and transform scale layout for DeepGEMM."""
    from ..tasks.baseline.L1.fp8_linear import Fp8Linear, postprocess_fp8_weights

    print("  Post-processing FP8 weights for DeepGEMM...")
    count = 0
    for module in model.modules():
        if isinstance(module.linear_op if hasattr(module, 'linear_op') else None, Fp8Linear):
            w = module.weight
            s = module.weight_scale_inv
            w_new, s_new = postprocess_fp8_weights(w.data, s.data)
            module.weight = torch.nn.Parameter(w_new, requires_grad=False)
            module.weight_scale_inv = torch.nn.Parameter(s_new, requires_grad=False)
            count += 1
    print(f"  Post-processed {count} FP8 linear layers.")


def _is_diffusion_model(model_name: str) -> bool:
    """Check if the model is a diffusion model (e.g., FLUX) by looking for model_index.json."""
    import json as _json
    model_path = model_name if os.path.isdir(model_name) else download_model(model_name)
    index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = _json.load(f)
        class_name = data.get("_class_name", "")
        if "Flux" in class_name or "Diffusion" in class_name:
            return True
    return False


def _detect_model_type(model_name: str) -> str:
    """Detect model architecture from HuggingFace config."""
    if _is_diffusion_model(model_name):
        return "flux"
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

    if model_type == "flux":
        raise ValueError(
            "FLUX diffusion models should be loaded via "
            "kb_nano.infra.diffusion_engine.DiffusionEngine, "
            "not the LLM load_model() path."
        )
    if model_type == "whisper":
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
    elif model_type == "qwen3_vl":
        config = Qwen3VLConfig.from_pretrained(model_name)
        config.dtype = dtype
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
        _postprocess_fp8_weights(model)
    else:
        model = model.to(device=device, dtype=dtype)

    model.eval()
    return model, config
