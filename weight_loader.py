"""
Weight loader for standalone Llama 3.1 / Mixtral with tensor parallelism.

Loads weights from HuggingFace safetensors and distributes them
across TP shards using the weight_loader callbacks on each parameter.
"""

from __future__ import annotations

import gc
import os
from glob import glob
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

from .tasks.L4.llama import LlamaConfig, LlamaForCausalLM
from .tasks.L4.mixtral import MixtralConfig, MixtralForCausalLM


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def download_model(model_name: str) -> str:
    return snapshot_download(
        model_name, allow_patterns=["*.safetensors", "*.json"],
    )


import re

_EXPERT_RE = re.compile(
    r"(.+\.block_sparse_moe)\.experts\.(\d+)\.(w[123])\.weight"
)


def load_weights(model: LlamaForCausalLM | MixtralForCausalLM, model_path: str) -> None:
    """Load weights with support for packed modules, MoE experts, and TP sharding.

    Handles:
    - q_proj/k_proj/v_proj -> qkv_proj (Llama and Mixtral)
    - gate_proj/up_proj -> gate_up_proj (Llama only)
    - experts.N.wX.weight -> stacked wX parameter with expert_id (Mixtral)
    Each parameter's weight_loader callback handles TP sharding.
    """
    packed = getattr(model, "packed_modules_mapping", {})
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    print(f"  Loading weights from {len(safetensor_files)} safetensors file(s)...")
    loaded = 0
    for sf_file in safetensor_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Handle MoE expert weights:
                #   experts.N.w1.weight -> w13 (first half)
                #   experts.N.w3.weight -> w13 (second half)
                #   experts.N.w2.weight -> w2
                m = _EXPERT_RE.match(weight_name)
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
                    if orig_key in weight_name:
                        param_name = weight_name.replace(orig_key, packed_name)
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
                if "rotary_emb" in weight_name:
                    continue
                try:
                    param = model.get_parameter(weight_name)
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


def load_model(
    model_name: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[LlamaForCausalLM | MixtralForCausalLM, LlamaConfig | MixtralConfig]:
    model_path = download_model(model_name)
    model_type = _detect_model_type(model_name)

    if model_type == "mixtral":
        config = MixtralConfig.from_pretrained(model_name)
        config.dtype = dtype
        print(f"  Allocating Mixtral model ({config.num_local_experts} experts)...")
        model = MixtralForCausalLM(config)
    else:
        config = LlamaConfig.from_pretrained(model_name)
        config.dtype = dtype
        print("  Allocating Llama model...")
        model = LlamaForCausalLM(config)

    load_weights(model, model_path)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config
