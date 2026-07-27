from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from fastkernels.tasks.baseline.L1.l2_norm import L2Norm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L3.xlm_roberta_model import XLMRobertaModel

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


@dataclass
class BGEM3Config:
    model_type: str = "xlm-roberta"
    vocab_size: int = 250002
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 8194
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    position_embedding_type: str = "absolute"
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "BGEM3Config":
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return cls(
            model_type=getattr(hf, "model_type", "xlm-roberta"),
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            intermediate_size=hf.intermediate_size,
            max_position_embeddings=hf.max_position_embeddings,
            type_vocab_size=hf.type_vocab_size,
            layer_norm_eps=hf.layer_norm_eps,
            hidden_act=hf.hidden_act,
            hidden_dropout_prob=hf.hidden_dropout_prob,
            attention_probs_dropout_prob=hf.attention_probs_dropout_prob,
            pad_token_id=hf.pad_token_id,
            bos_token_id=getattr(hf, "bos_token_id", 0),
            eos_token_id=getattr(hf, "eos_token_id", 2),
            position_embedding_type=getattr(hf, "position_embedding_type", "absolute"),
        )


if _HAS_TRITON:

    @triton.jit
    def _l2_norm_mask_kernel(
        x_ptr,
        mask_ptr,
        y_ptr,
        n_cols: tl.constexpr,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        m_s0: tl.constexpr,
        m_s1: tl.constexpr,
        y_s0: tl.constexpr,
        y_s1: tl.constexpr,
        y_s2: tl.constexpr,
        t_cols: tl.constexpr,
        has_mask: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // t_cols
        t = row - b * t_cols
        offs = tl.arange(0, block)
        valid = offs < n_cols
        vals = tl.load(x_ptr + b * x_s0 + t * x_s1 + offs * x_s2, mask=valid, other=0.0).to(tl.float32)
        if has_mask:
            m = tl.load(mask_ptr + b * m_s0 + t * m_s1).to(tl.float32)
            vals *= m
        ss = tl.sum(vals * vals, axis=0)
        denom = tl.sqrt(ss)
        inv = tl.where(denom > 1.0e-12, 1.0 / denom, 1.0e12)
        vals *= inv
        tl.store(y_ptr + b * y_s0 + t * y_s1 + offs * y_s2, vals, mask=valid)

    @triton.jit
    def _classify_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        mask_ptr,
        y_ptr,
        n_cols: tl.constexpr,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        m_s0: tl.constexpr,
        m_s1: tl.constexpr,
        s_cols: tl.constexpr,
        has_bias: tl.constexpr,
        has_mask: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // s_cols
        s = row - b * s_cols
        offs = tl.arange(0, block)
        valid = offs < n_cols
        vals = tl.load(x_ptr + b * x_s0 + s * x_s1 + offs * x_s2, mask=valid, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=valid, other=0.0).to(tl.float32)
        acc = tl.sum(vals * w, axis=0)
        if has_bias:
            acc += tl.load(b_ptr).to(tl.float32)
        acc = tl.maximum(acc, 0.0)
        if has_mask:
            acc *= tl.load(mask_ptr + b * m_s0 + s * m_s1).to(tl.float32)
        tl.store(y_ptr + row, acc)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _num_warps(block: int) -> int:
    if block >= 2048:
        return 8
    if block >= 512:
        return 4
    return 1


def _fallback_l2_norm(vecs: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None:
        vecs = vecs * attention_mask.unsqueeze(-1).to(dtype=vecs.dtype)
    return F.normalize(vecs, p=2.0, dim=-1, eps=1.0e-12)


def _fast_l2_norm(vecs: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if (
        not _HAS_TRITON
        or not vecs.is_cuda
        or vecs.dim() != 3
        or vecs.shape[-1] <= 0
        or vecs.shape[-1] > 8192
    ):
        return _fallback_l2_norm(vecs, attention_mask)

    bsz, t_cols, n_cols = vecs.shape
    if bsz == 0 or t_cols == 0:
        return torch.empty_like(vecs)
    block = _next_power_of_2(n_cols)
    out = torch.empty((bsz, t_cols, n_cols), device=vecs.device, dtype=vecs.dtype)
    mask_arg = attention_mask if attention_mask is not None else vecs
    m_s0 = mask_arg.stride(0) if attention_mask is not None else 0
    m_s1 = mask_arg.stride(1) if attention_mask is not None else 0
    _l2_norm_mask_kernel[(bsz * t_cols,)](
        vecs,
        mask_arg,
        out,
        n_cols,
        vecs.stride(0),
        vecs.stride(1),
        vecs.stride(2),
        m_s0,
        m_s1,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        t_cols,
        attention_mask is not None,
        block,
        num_warps=_num_warps(block),
    )
    return out


def _fallback_classify(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    weights = F.relu(F.linear(hidden_states, weight, bias))
    if attention_mask is not None:
        weights = weights * attention_mask.unsqueeze(-1).to(dtype=weights.dtype)
    return weights


def _fast_classify(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if (
        not _HAS_TRITON
        or not hidden_states.is_cuda
        or hidden_states.dim() != 3
        or weight.dim() != 2
        or weight.shape[0] != 1
        or hidden_states.shape[-1] != weight.shape[1]
        or hidden_states.shape[-1] <= 0
        or hidden_states.shape[-1] > 8192
    ):
        return _fallback_classify(hidden_states, weight, bias, attention_mask)

    bsz, s_cols, n_cols = hidden_states.shape
    if bsz == 0 or s_cols == 0:
        return torch.empty((bsz, s_cols, 1), device=hidden_states.device, dtype=weight.dtype)
    block = _next_power_of_2(n_cols)
    out = torch.empty((bsz, s_cols, 1), device=hidden_states.device, dtype=weight.dtype)
    mask_arg = attention_mask if attention_mask is not None else hidden_states
    m_s0 = mask_arg.stride(0) if attention_mask is not None else 0
    m_s1 = mask_arg.stride(1) if attention_mask is not None else 0
    bias_arg = bias if bias is not None else weight
    _classify_kernel[(bsz * s_cols,)](
        hidden_states,
        weight,
        bias_arg,
        mask_arg,
        out,
        n_cols,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        m_s0,
        m_s1,
        s_cols,
        bias is not None,
        attention_mask is not None,
        block,
        num_warps=_num_warps(block),
    )
    return out


class BgeM3EmbeddingModel(nn.Module):
    is_pooling_model = True

    def __init__(self, config: BGEM3Config):
        super().__init__()
        self.config = config
        self.model = XLMRobertaModel(config)
        self.sparse_linear = Linear(config.hidden_size, 1, bias=True)
        self.colbert_linear = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.relu = ReLU()
        self.norm = L2Norm(dim=-1)
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def forward_with_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.forward_with_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )

    def forward_varlen(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.forward_varlen(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def token_embed(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_dtype = self.colbert_linear.weight.dtype
        hidden_states = hidden_states if hidden_states.dtype == target_dtype else hidden_states.to(target_dtype)
        tail = hidden_states[:, 1:]
        if hidden_states.dim() == 3 and hidden_states.shape[1] >= 16 and not tail.is_contiguous():
            vecs = F.linear(hidden_states, self.colbert_linear.weight, self.colbert_linear.bias)[:, 1:]
        else:
            vecs = F.linear(tail, self.colbert_linear.weight, self.colbert_linear.bias)
        mask = attention_mask[:, 1:] if attention_mask is not None else None
        return _fast_l2_norm(vecs, mask)

    def token_classify(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_dtype = self.sparse_linear.weight.dtype
        hidden_states = hidden_states if hidden_states.dtype == target_dtype else hidden_states.to(target_dtype)
        return _fast_classify(hidden_states, self.sparse_linear.weight, self.sparse_linear.bias, attention_mask)
