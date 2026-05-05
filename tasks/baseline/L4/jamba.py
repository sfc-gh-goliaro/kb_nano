"""Standalone Jamba model implementation (L4 pipeline wiring).

Jamba (AI21Labs) is a triple-hybrid:

  * **Transformer attention** in a small fraction of layers (4 out of
    every 32 in v0.1, 4 out of every 16 in tiny-dev -- the schedule
    derives from ``attn_layer_period``/``attn_layer_offset`` in the HF
    config).
  * **Mamba v1** selective state-space mixers in the remaining majority
    of layers.
  * **Sparse Mixture of Experts** (top-2 of 16 in v0.1, top-2 of 8 in
    tiny-dev) on every other layer's FFN.  The non-MoE layers use a
    plain SwiGLU MLP.

The mix gives Jamba large effective context with subquadratic compute:
the few attention layers carry global token-token mixing, the Mamba
layers carry the bulk of the work in linear time, and MoE adds
parameter capacity without proportional FLOPs.

This file is the L4 pipeline wiring: layer construction, embedding,
final layer norm, LM head, and a checkpoint-aligned ``forward``.  All
heavy lifting lives in the L2/L3 modules.

Reference for the layer schedule and weight names:
``transformers.models.jamba.modeling_jamba`` (HuggingFace).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L1.rms_norm import RMSNorm
from ..L3.jamba_decoder import (
    JambaAttentionDecoderLayer,
    JambaMambaDecoderLayer,
)


# ---------------------------------------------------------------------------
# Config (mirrors transformers.models.jamba.configuration_jamba.JambaConfig
# field-for-field for the bits the pipeline actually consumes).
# ---------------------------------------------------------------------------
@dataclass
class JambaConfig:
    # ---- Common LLM fields ----
    vocab_size: int = 65536
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False

    # ---- Mamba mixer fields ----
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int = 256
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    # ---- MoE fields ----
    num_experts: int = 16
    num_experts_per_tok: int = 2

    # ---- Hybrid recipe ----
    # Attention layers at indices ``i % attn_layer_period == attn_layer_offset``.
    # MoE layers   at indices ``i % expert_layer_period == expert_layer_offset``.
    attn_layer_period: int = 8
    attn_layer_offset: int = 4
    expert_layer_period: int = 2
    expert_layer_offset: int = 1

    # Derived: per-layer block type and per-layer expert count.
    # Filled by ``__post_init__`` if not provided explicitly.
    layers_block_type: list[str] = field(default_factory=list)
    layers_num_experts: list[int] = field(default_factory=list)

    # Inference dtype.
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if not self.layers_block_type:
            self.layers_block_type = [
                "attention" if (i % self.attn_layer_period == self.attn_layer_offset)
                else "mamba"
                for i in range(self.num_hidden_layers)
            ]
        if not self.layers_num_experts:
            self.layers_num_experts = [
                self.num_experts
                if (i % self.expert_layer_period == self.expert_layer_offset)
                else 1
                for i in range(self.num_hidden_layers)
            ]
        assert len(self.layers_block_type) == self.num_hidden_layers
        assert len(self.layers_num_experts) == self.num_hidden_layers

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "JambaConfig":
        path = Path(model_path)
        if path.is_dir():
            config_path = path / "config.json"
        else:
            config_path = path
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        kwargs = {
            "vocab_size": data["vocab_size"],
            "hidden_size": data["hidden_size"],
            "intermediate_size": data["intermediate_size"],
            "num_hidden_layers": data["num_hidden_layers"],
            "num_attention_heads": data["num_attention_heads"],
            "num_key_value_heads": data["num_key_value_heads"],
            "rms_norm_eps": data.get("rms_norm_eps", 1e-6),
            "pad_token_id": data.get("pad_token_id", 0),
            "bos_token_id": data.get("bos_token_id", 1),
            "eos_token_id": data.get("eos_token_id", 2),
            "tie_word_embeddings": data.get("tie_word_embeddings", False),
            "mamba_d_state": data.get("mamba_d_state", 16),
            "mamba_d_conv": data.get("mamba_d_conv", 4),
            "mamba_expand": data.get("mamba_expand", 2),
            "mamba_dt_rank": data.get("mamba_dt_rank", 256),
            "mamba_conv_bias": data.get("mamba_conv_bias", True),
            "mamba_proj_bias": data.get("mamba_proj_bias", False),
            "num_experts": data.get("num_experts", 16),
            "num_experts_per_tok": data.get("num_experts_per_tok", 2),
            "attn_layer_period": data.get("attn_layer_period", 8),
            "attn_layer_offset": data.get("attn_layer_offset", 4),
            "expert_layer_period": data.get("expert_layer_period", 2),
            "expert_layer_offset": data.get("expert_layer_offset", 1),
        }
        # Some checkpoints (e.g. v0.1) ship the explicit per-layer
        # arrays; honour those when present so we never disagree with
        # the checkpoint about which layer is which.
        if "layers_block_type" in data:
            kwargs["layers_block_type"] = list(data["layers_block_type"])
        if "layers_num_experts" in data:
            kwargs["layers_num_experts"] = list(data["layers_num_experts"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class JambaModel(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id,
        )

        layers: list[nn.Module] = []
        for i in range(config.num_hidden_layers):
            block_type = config.layers_block_type[i]
            if block_type == "attention":
                layers.append(JambaAttentionDecoderLayer(config, layer_idx=i))
            elif block_type == "mamba":
                layers.append(JambaMambaDecoderLayer(config, layer_idx=i))
            else:  # pragma: no cover -- defensive
                raise ValueError(f"Unknown layers_block_type[{i}]={block_type!r}")
        self.layers = nn.ModuleList(layers)

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Convenience handle: indices of the attention vs mamba layers.
        # Used by the engine to size KV cache slabs and to build the
        # layered cache views.
        self.attention_layer_indices = [
            i for i, t in enumerate(config.layers_block_type) if t == "attention"
        ]
        self.mamba_layer_indices = [
            i for i, t in enumerate(config.layers_block_type) if t == "mamba"
        ]

    def forward(
        self,
        input_ids: torch.Tensor,                                         # [B, T]
        attn_past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        attn_cache_writeback: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        attn_mask_4d: torch.Tensor | None = None,
        # Static-shape decode path (CUDA-graph friendly).  When
        # ``attn_kv_slabs`` is provided, the per-layer KV is held in a
        # fixed-shape ``[B, H_kv, max_total, D]`` slab and the new
        # token's K/V are written at ``attn_slot_pos`` via ``index_copy_``.
        # Mutually exclusive with ``attn_past_kv``/``attn_cache_writeback``.
        attn_kv_slabs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attn_slot_pos: torch.Tensor | None = None,
        # Mamba-side cache + metadata.  Each is a *list* parallel to
        # ``mamba_layer_indices`` so per-layer state is independent.
        mamba_conv_states: list[torch.Tensor] | None = None,
        mamba_ssm_states: list[torch.Tensor] | None = None,
        mamba_cache_indices: torch.Tensor | None = None,
        mamba_query_start_loc: torch.Tensor | None = None,
        mamba_has_initial_state: torch.Tensor | None = None,
        mamba_is_decode: bool = False,
        mamba_pad_mask_flat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list]:
        """Forward.

        Parameters mirror the structure HF uses, but specialised for our
        engine which holds the caches externally.

        Returns
        -------
        (hidden_states, attn_kv_outputs)
            ``attn_kv_outputs`` is a list, parallel to
            ``attention_layer_indices``, of (k, v) tensors that include
            the new tokens (caller decides whether to keep them).
        """
        hidden_states = self.embed_tokens(input_ids)

        if attn_past_kv is None:
            attn_past_kv = [None] * len(self.attention_layer_indices)
        if attn_cache_writeback is None:
            attn_cache_writeback = [None] * len(self.attention_layer_indices)

        attn_kv_outputs: list = []
        attn_idx = 0
        mamba_idx = 0

        for layer in self.layers:
            if isinstance(layer, JambaAttentionDecoderLayer):
                if attn_kv_slabs is not None:
                    # Static-shape decode path.
                    k_slab, v_slab = attn_kv_slabs[attn_idx]
                    hidden_states, new_k, new_v = layer(
                        hidden_states,
                        attention_mask=attn_mask_4d,
                        kv_slab=(k_slab, v_slab),
                        slot_pos=attn_slot_pos,
                    )
                else:
                    pkv = attn_past_kv[attn_idx]
                    writeback = attn_cache_writeback[attn_idx]
                    past_k = pkv[0] if pkv is not None else None
                    past_v = pkv[1] if pkv is not None else None
                    hidden_states, new_k, new_v = layer(
                        hidden_states,
                        attention_mask=attn_mask_4d,
                        past_key=past_k,
                        past_value=past_v,
                        cache_writeback=writeback,
                    )
                attn_kv_outputs.append((new_k, new_v))
                attn_idx += 1
            else:
                conv_state = mamba_conv_states[mamba_idx]
                ssm_state = mamba_ssm_states[mamba_idx]
                hidden_states = layer(
                    hidden_states,
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                    cache_indices=mamba_cache_indices,
                    query_start_loc=mamba_query_start_loc,
                    has_initial_state=mamba_has_initial_state,
                    is_decode=mamba_is_decode,
                    mamba_pad_mask_flat=mamba_pad_mask_flat,
                )
                mamba_idx += 1

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, attn_kv_outputs


class JambaForCausalLM(nn.Module):
    """Jamba LM head wrapper.

    The HF checkpoint stores the LM head separately
    (``tie_word_embeddings=False``) -- we load both ``embed_tokens`` and
    ``lm_head`` from the safetensors.

    Weight name layout (matches HF transformers' Jamba checkpoint):

        model.embed_tokens.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.pre_ff_layernorm.weight

        # Attention layers:
        model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight

        # Mamba layers:
        model.layers.{i}.mamba.in_proj.weight
        model.layers.{i}.mamba.conv1d.weight        [intermediate, 1, K]
        model.layers.{i}.mamba.conv1d.bias          [intermediate]
        model.layers.{i}.mamba.x_proj.weight
        model.layers.{i}.mamba.dt_proj.weight
        model.layers.{i}.mamba.dt_proj.bias
        model.layers.{i}.mamba.A_log
        model.layers.{i}.mamba.D
        model.layers.{i}.mamba.out_proj.weight
        model.layers.{i}.mamba.dt_layernorm.weight
        model.layers.{i}.mamba.b_layernorm.weight
        model.layers.{i}.mamba.c_layernorm.weight

        # FFN (MLP layers):
        model.layers.{i}.feed_forward.{gate_proj,up_proj,down_proj}.weight

        # FFN (MoE layers):
        model.layers.{i}.feed_forward.router.weight
        model.layers.{i}.feed_forward.experts.{e}.{gate_proj,up_proj,down_proj}.weight

        model.final_layernorm.weight
        lm_head.weight
    """

    def __init__(self, config: JambaConfig):
        super().__init__()
        self.config = config
        self.model = JambaModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.emb.weight

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits.float()

    # ------------------------------------------------------------------
    # Weight loading: read the HF safetensors and dispatch to per-param
    # weight_loader callbacks (used for the MoE expert pack-into-3D-tensor
    # and the Mamba A_log -> A negative-exp transform).  Mirrors the
    # ``Pattern 2`` per-pipeline loader documented in CLAUDE.md, since
    # Jamba's hybrid wiring needs custom routing.
    # ------------------------------------------------------------------
    def load_weights(self, model_path: str) -> int:
        """Load Jamba weights from a local snapshot directory.

        Returns the number of tensors copied into model parameters.
        """
        import re
        from glob import glob
        from safetensors import safe_open

        config = self.config

        # Build a name -> Parameter map for direct keys.  MoE expert
        # weights need pattern matching, so handle them in the loop.
        params = dict(self.named_parameters())

        moe_re = re.compile(
            r"^model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)\.weight$"
        )

        sf_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors in {model_path}")

        loaded = 0
        unmatched: list[str] = []
        for sf in sf_files:
            with safe_open(sf, "pt", "cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    mapped = self._remap_name(name)
                    if mapped is None:
                        unmatched.append(name)
                        continue
                    # MoE expert: pack into 3D tensor at (expert_idx, ...)
                    m = moe_re.match(name)
                    if m is not None:
                        layer_idx = int(m.group(1))
                        expert_idx = int(m.group(2))
                        proj = m.group(3)
                        ff_path = f"model.layers.{layer_idx}.feed_forward"
                        if proj == "gate_proj":
                            param = params[f"{ff_path}.w13"]
                            param.weight_loader(param, tensor, expert_idx, True)
                        elif proj == "up_proj":
                            param = params[f"{ff_path}.w13"]
                            param.weight_loader(param, tensor, expert_idx, False)
                        else:
                            param = params[f"{ff_path}.w2"]
                            param.weight_loader(param, tensor, expert_idx)
                        loaded += 1
                        continue

                    # Direct copy / A_log transform.
                    if mapped not in params:
                        unmatched.append(name)
                        continue
                    param = params[mapped]
                    if hasattr(param, "weight_loader"):
                        param.weight_loader(param, tensor)
                    else:
                        if tensor.shape != param.data.shape:
                            raise RuntimeError(
                                f"shape mismatch loading {name} "
                                f"(checkpoint {tuple(tensor.shape)} vs "
                                f"model {tuple(param.data.shape)})"
                            )
                        param.data.copy_(tensor)
                    loaded += 1

        if unmatched:
            # Be loud about anything we silently dropped -- helps catch
            # a checkpoint vs config mismatch (e.g. expert count).
            print(f"  [JambaForCausalLM] WARNING: {len(unmatched)} unmatched "
                  f"weight names; first few: {unmatched[:5]}")
        return loaded

    @staticmethod
    def _remap_name(name: str) -> str | None:
        """HF checkpoint -> our param path.  Returns None if MoE expert
        (handled separately) or a name we should silently skip."""
        # Embedding wrapper: HF stores at model.embed_tokens.weight,
        # our L1 ``Embedding`` nests an nn.Embedding as ``self.emb``.
        if name == "model.embed_tokens.weight":
            return "model.embed_tokens.emb.weight"
        if name == "lm_head.weight":
            return "lm_head.weight"
        if name == "model.final_layernorm.weight":
            return "model.final_layernorm.weight"

        # Mamba-mixer: A_log -> A (param has its own weight_loader to
        # apply -exp() at load time), conv1d.weight -> conv1d_weight,
        # conv1d.bias -> conv1d_bias.
        if "mamba.A_log" in name:
            return name.replace("mamba.A_log", "mamba.A")
        if "mamba.conv1d.weight" in name:
            return name.replace("mamba.conv1d.weight", "mamba.conv1d_weight")
        if "mamba.conv1d.bias" in name:
            return name.replace("mamba.conv1d.bias", "mamba.conv1d_bias")

        # MoE expert weights are packed via the param's weight_loader,
        # not a direct copy -- signal that by returning ``None`` so the
        # caller skips the direct-copy path.  We still reach the MoE
        # handling because it uses the original (un-mapped) name.
        if ".feed_forward.experts." in name:
            return name  # placeholder; caller checks moe_re

        # Everything else is a 1-to-1 mapping.
        return name
