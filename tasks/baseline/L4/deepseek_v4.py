"""DeepSeek-V4-Flash pipeline (L4) for kb_nano / fastkernels.

DeepSeek V4 is architecturally distinct from V2/V3/V3.2:

* **Hyper-connections (mHC)** replace the residual stream — the hidden state is
  widened to ``hc_mult`` parallel streams mixed by Sinkhorn-normalized routing
  at every sub-block (see ``..L1.mhc``).
* **MLA with attention sink + per-layer compression** — each layer has a
  ``compress_ratio`` in {1, 4, 128} selecting full sliding-window attention
  (SWA), C4A sparse (Lightning indexer) or C128A compressed attention, all on
  an ``fp8_ds_mla`` paged cache with a 576B-aligned layout, plus a learned
  attention sink and an output low-rank (``o_lora_rank`` / ``o_groups``) path.
* **MXFP4 routed experts** with ``sqrtsoftplus`` scoring, ``noaux_tc`` bias and
  hash-MoE routing on the first ``num_hash_layers`` layers.

Per the chosen strategy, the heavy kernels are **reused verbatim** from the
installed vLLM 0.20.0 wheel (FlashMLA-sparse, vendored DeepGEMM FP8/FP4,
TileLang mHC, MXFP4 MoE).  This module owns the kb_nano-side config object; the
decoder/attention/MoE live in L3/L2 and the engine bridge in
``infra.deepseek_v4_engine``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DeepSeekV4Config:
    """kb_nano-side view of the DeepSeek-V4-Flash HF config.

    Only the fields the engine + model modules consult are surfaced as
    attributes; the full HF config dict is retained in ``hf`` for the vLLM
    module construction path.
    """

    hidden_size: int = 4096
    intermediate_size: int = 2048
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    vocab_size: int = 129280
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0

    # MLA / V4 attention geometry
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    compress_ratios: list[int] = field(default_factory=list)

    # Lightning indexer (C4A layers)
    index_topk: int = 512
    index_n_heads: int = 64
    index_head_dim: int = 128

    # MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    hidden_act: str = "silu"

    # Hyper-connections
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20

    # MTP (draft) layers — skipped for non-speculative serving
    num_nextn_predict_layers: int = 1

    rope_scaling: dict = field(default_factory=dict)
    quantization_config: dict = field(default_factory=dict)

    dtype: torch.dtype = torch.bfloat16
    hf: Any = None
    # Set by the engine before model construction so MLA/indexer buffers are
    # sized to the runtime batch budget.
    max_num_batched_tokens: int = 16384

    @classmethod
    def from_pretrained(cls, model_name: str) -> "DeepSeekV4Config":
        if os.path.isdir(model_name):
            cfg_path = os.path.join(model_name, "config.json")
        else:
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(model_name, "config.json")
        with open(cfg_path) as f:
            c = json.load(f)

        return cls(
            hidden_size=c["hidden_size"],
            intermediate_size=c.get("intermediate_size", c["moe_intermediate_size"]),
            moe_intermediate_size=c["moe_intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", 1),
            vocab_size=c["vocab_size"],
            max_position_embeddings=c["max_position_embeddings"],
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=c.get("rope_theta", 10000.0),
            compress_rope_theta=c.get("compress_rope_theta", 160000.0),
            head_dim=c["head_dim"],
            qk_rope_head_dim=c["qk_rope_head_dim"],
            q_lora_rank=c["q_lora_rank"],
            o_lora_rank=c["o_lora_rank"],
            o_groups=c["o_groups"],
            sliding_window=c["sliding_window"],
            compress_ratios=list(c.get("compress_ratios", [])),
            index_topk=c.get("index_topk", 512),
            index_n_heads=c.get("index_n_heads", 64),
            index_head_dim=c.get("index_head_dim", 128),
            n_routed_experts=c["n_routed_experts"],
            n_shared_experts=c.get("n_shared_experts", 1),
            num_experts_per_tok=c["num_experts_per_tok"],
            num_hash_layers=c.get("num_hash_layers", 0),
            routed_scaling_factor=c.get("routed_scaling_factor", 1.0),
            scoring_func=c.get("scoring_func", "sqrtsoftplus"),
            topk_method=c.get("topk_method", "noaux_tc"),
            norm_topk_prob=c.get("norm_topk_prob", True),
            swiglu_limit=c.get("swiglu_limit", 10.0),
            hidden_act=c.get("hidden_act", "silu"),
            hc_mult=c.get("hc_mult", 4),
            hc_eps=c.get("hc_eps", 1e-6),
            hc_sinkhorn_iters=c.get("hc_sinkhorn_iters", 20),
            num_nextn_predict_layers=c.get("num_nextn_predict_layers", 1),
            rope_scaling=c.get("rope_scaling", {}) or {},
            quantization_config=c.get("quantization_config", {}) or {},
            hf=None,
        )

    # NOTE: V4 deliberately does NOT expose ``kv_lora_rank``. The engine keys
    # its legacy V2/V3.2 MLA path off ``hasattr(config, "kv_lora_rank")``; V4
    # has a structurally different attention (sparse-SWA + compression + attn
    # sink) and is routed by ``model_type == "deepseek_v4"`` to a dedicated
    # engine bridge instead.
    @property
    def model_type(self) -> str:
        return "deepseek_v4"
