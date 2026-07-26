"""BGE-M3 vLLM-compatible embedding model wiring."""


from __future__ import annotations


# Inlined from tasks/reference/L1/l2_norm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)


# Inlined from tasks/reference/L1/linear.py
class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


# Inlined from tasks/reference/L1/relu.py
class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


# Inlined from tasks/reference/L1/embedding.py
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=padding_idx)

    def forward(self, input_ids):
        return self.emb(input_ids)


# Inlined from tasks/reference/L1/layer_norm.py
class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)
        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)


# Inlined from tasks/reference/L2/encoder_embeddings.py
TOKEN_TYPE_SHIFT = 30


def encode_token_type_ids(input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> None:
    input_ids[: token_type_ids.shape[0]].bitwise_or_(token_type_ids << TOKEN_TYPE_SHIFT)


def decode_token_type_ids(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids_mask = (
        torch.ones_like(input_ids, dtype=torch.int32, device=input_ids.device)
        << TOKEN_TYPE_SHIFT
    )
    tokens_mask = ids_mask.bitwise_not()
    token_type_ids = input_ids.bitwise_and(ids_mask) >> TOKEN_TYPE_SHIFT
    return input_ids.bitwise_and(tokens_mask), token_type_ids


def create_roberta_position_ids_from_input_ids(
    input_ids: torch.Tensor,
    padding_idx: int,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    mask = input_ids.ne(padding_idx).int()
    incremental = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental.long() + padding_idx


class EncoderEmbeddingsBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        word_padding_idx = self._word_embedding_padding_idx(config)
        position_padding_idx = self._position_embedding_padding_idx(config)

        self.word_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=word_padding_idx,
        )
        self.position_embeddings = Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=position_padding_idx,
        )
        self.token_type_embeddings = Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.LayerNorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def _word_embedding_padding_idx(self, config) -> int | None:
        return None

    def _position_embedding_padding_idx(self, config) -> int | None:
        return None

    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        past_key_values_length: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids, token_type_ids = decode_token_type_ids(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = (
            inputs_embeds
            + self.token_type_embeddings(token_type_ids.to(device=input_ids.device))
        )
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids.to(device=input_ids.device))
        return self.LayerNorm(embeddings)

    def forward_with_token_type_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, : input_ids.size(1)].expand(input_ids.size())
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids.to(input_ids.device))
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids.to(input_ids.device))
        return self.LayerNorm(embeddings)


class BertEmbeddings(EncoderEmbeddingsBase):
    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            seq_len = input_ids.size(1)
            return self.position_ids[
                :,
                past_key_values_length: past_key_values_length + seq_len,
            ]

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided when input_ids is None")
        input_shape = inputs_embeds.size()[:-1]
        seq_len = input_shape[1]
        return self.position_ids[
            :,
            past_key_values_length: past_key_values_length + seq_len,
        ]


class XLMRobertaEmbeddings(EncoderEmbeddingsBase):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id

    def _word_embedding_padding_idx(self, config) -> int | None:
        return config.pad_token_id

    def _position_embedding_padding_idx(self, config) -> int | None:
        return config.pad_token_id

    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            return create_roberta_position_ids_from_input_ids(
                input_ids,
                self.padding_idx,
                past_key_values_length,
            )

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided when input_ids is None")
        input_shape = inputs_embeds.size()[:-1]
        seq_len = input_shape[1]
        return torch.arange(
            self.padding_idx + 1,
            seq_len + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        ).unsqueeze(0).expand(input_shape)


# Inlined from tasks/reference/L1/dense_attention.py
from typing import Literal


class DenseAttention(nn.Module):
    """Dense multi-head attention with ``(batch, seq, heads, dim)`` layout."""

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn"] = "auto"):
        super().__init__()
        del backend

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask=None,
    ):
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.permute(0, 2, 1, 3)


# Inlined helper (no baseline task): pure-torch dense/varlen attention
# and paged-cache gather.  The baselines call flash_attn / vllm_flash_attn
# here, so there is no baseline file to inline this from.


def repeat_kv(k: torch.Tensor, target_heads: int) -> torch.Tensor:
    if k.shape[-2] == target_heads:
        return k
    if target_heads % k.shape[-2] != 0:
        raise ValueError(
            f"Cannot repeat {k.shape[-2]} KV heads to {target_heads} query heads"
        )
    return k.repeat_interleave(target_heads // k.shape[-2], dim=-2)


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] | list[int] | None = (-1, -1),
    s_aux: torch.Tensor | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    window_size = (-1, -1) if window_size is None else tuple(window_size)
    q_in = q.transpose(-3, -2)
    k_in = repeat_kv(k, q.shape[-2]).transpose(-3, -2)
    v_in = repeat_kv(v, q.shape[-2]).transpose(-3, -2)
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    has_backend_specific_mask = (
        window_size != (-1, -1)
        or s_aux is not None
        or softcap > 0.0
    )
    if q.is_cuda and not has_backend_specific_mask and q_in.shape[-2] == k_in.shape[-2]:
        out = torch.ops.aten._scaled_dot_product_flash_attention(
            q_in, k_in, v_in, 0.0, causal, scale=scale,
        )[0]
        return out.transpose(-3, -2)
    if (
        q.is_cuda
        and causal
        and not has_backend_specific_mask
        and q_in.shape[-2] == 1
    ):
        out = torch.ops.aten._scaled_dot_product_flash_attention(
            q_in, k_in, v_in, 0.0, False, scale=scale,
        )[0]
        return out.transpose(-3, -2)
    if causal or has_backend_specific_mask:
        q_len = q_in.shape[-2]
        k_len = k_in.shape[-2]
        left, right = window_size
        if causal:
            right = 0
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1) + (k_len - q_len)
        k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
        if left < 0:
            mask = k_pos <= q_pos + right
        else:
            mask = (k_pos <= torch.minimum(q_pos + right, torch.full_like(q_pos, k_len))) & (
                k_pos >= q_pos - left
            )
        scores = torch.matmul(q_in.float(), k_in.float().transpose(-2, -1)) * scale
        if softcap > 0.0:
            scores = torch.tanh(scores / softcap) * softcap
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        if s_aux is not None:
            sink = s_aux.to(device=scores.device, dtype=scores.dtype).view(1, -1, 1, 1)
            sink = sink.expand(scores.shape[0], -1, scores.shape[-2], -1)
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1)
        probs = probs.masked_fill(torch.all(~mask, dim=-1, keepdim=True), 0.0)
        if s_aux is not None:
            out = torch.matmul(probs, v_in.float()).to(v_in.dtype)
        else:
            out = torch.matmul(probs.to(v_in.dtype), v_in)
        return out.transpose(-3, -2)
    out = F.scaled_dot_product_attention(
        q_in, k_in, v_in, is_causal=False, scale=scale,
    )
    return out.transpose(-3, -2)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] | list[int] | None = (-1, -1),
    s_aux: torch.Tensor | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    window_size = (-1, -1) if window_size is None else tuple(window_size)
    outputs = []
    batch = cu_seqlens_q.numel() - 1
    for i in range(batch):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())
        out = dense_attention(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            v[k_start:k_end].unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            s_aux=s_aux,
            softcap=softcap,
        ).squeeze(0)
        outputs.append(out)
    if not outputs:
        return q.new_empty(q.shape)
    return torch.cat(outputs, dim=0)


def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor | None,
    seq_idx: int,
    seq_len: int,
    *,
    hnd: bool = False,
) -> torch.Tensor:
    if block_table is None:
        if cache.ndim == 4 and hnd:
            return cache.reshape(-1, cache.shape[1], cache.shape[-1])[:seq_len]
        if cache.ndim == 4:
            return cache.reshape(-1, cache.shape[-2], cache.shape[-1])[:seq_len]
        return cache[:seq_len]

    blocks = block_table[seq_idx]
    pieces = []
    remaining = seq_len
    for block in blocks:
        if remaining <= 0:
            break
        block_idx = int(block.item())
        if block_idx < 0:
            continue
        block_cache = cache[block_idx]
        if hnd:
            block_cache = block_cache.transpose(0, 1)
        take = min(remaining, block_cache.shape[0])
        pieces.append(block_cache[:take])
        remaining -= take
    if not pieces:
        shape = (0, cache.shape[1], cache.shape[-1]) if hnd else (0, cache.shape[-2], cache.shape[-1])
        return cache.new_empty(shape)
    return torch.cat(pieces, dim=0)


def _varlen_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    batch = cu_seqlens_q.numel() - 1
    num_heads = q.shape[1]
    max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()) if batch else 0
    lse = torch.full((batch, num_heads, max_q), -float("inf"), dtype=torch.float32, device=q.device)
    for b in range(batch):
        qs = int(cu_seqlens_q[b].item())
        qe = int(cu_seqlens_q[b + 1].item())
        ks = int(cu_seqlens_k[b].item())
        ke = int(cu_seqlens_k[b + 1].item())
        q_b = q[qs:qe].float().transpose(0, 1)
        k_b = k[ks:ke].float().transpose(0, 1)
        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * softmax_scale
        if causal:
            sq = qe - qs
            sk = ke - ks
            q_pos = torch.arange(sq, device=q.device) + max(sk - sq, 0)
            k_pos = torch.arange(sk, device=q.device)
            mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            scores = scores.masked_fill(mask.unsqueeze(0), -float("inf"))
        lse[b, :, : qe - qs] = torch.logsumexp(scores, dim=-1)
    return lse


# Inlined from tasks/reference/L1/flash_attn_varlen.py
class FlashAttnVarlen(nn.Module):
    """Variable-length attention without paged KV cache lookup."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        del max_seqlen_q, max_seqlen_k
        out = varlen_attention(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            causal,
        )
        if not return_softmax_lse:
            return out
        return out, _varlen_lse(q, k, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal)


# Inlined from tasks/baseline/L2/encoder_attention.py
class EncoderSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"num_attention_heads={config.num_attention_heads}",
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key = Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value = Linear(config.hidden_size, self.all_head_size, bias=True)
        self.attn = DenseAttention(backend="sdpa")
        self.varlen_attn = FlashAttnVarlen()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_attention_mask(hidden_states)

    def forward_with_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.query(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )
        key = self.key(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )
        value = self.value(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )
        context = self.attn(query, key, value, causal=False, attn_mask=attention_mask)
        return context.contiguous().view(batch_size, seq_len, self.all_head_size)

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        query = self.query(hidden_states).view(
            hidden_states.size(0),
            self.num_attention_heads,
            self.attention_head_size,
        )
        key = self.key(hidden_states).view(
            hidden_states.size(0),
            self.num_attention_heads,
            self.attention_head_size,
        )
        value = self.value(hidden_states).view(
            hidden_states.size(0),
            self.num_attention_heads,
            self.attention_head_size,
        )
        context = self.varlen_attn(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.attention_head_size ** -0.5,
            causal=False,
        )
        if isinstance(context, tuple):
            context = context[0]
        return context.contiguous().view(hidden_states.size(0), self.all_head_size)


class EncoderSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class EncoderAttention(nn.Module):
    self_attention_cls = EncoderSelfAttention
    self_output_cls = EncoderSelfOutput

    def __init__(self, config):
        super().__init__()
        self.self = self.self_attention_cls(config)
        self.output = self.self_output_cls(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.self(hidden_states)
        return self.output(attention_output, hidden_states)

    def forward_with_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.self.forward_with_attention_mask(
            hidden_states,
            attention_mask=attention_mask,
        )
        return self.output(attention_output, hidden_states)

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        attention_output = self.self.forward_varlen(hidden_states, cu_seqlens, max_seqlen)
        return self.output(attention_output, hidden_states)


# Inlined from tasks/reference/L1/gelu.py
class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


# Inlined from tasks/reference/L2/encoder_mlp.py
class EncoderIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class EncoderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


# Inlined from tasks/baseline/L3/xlm_roberta_layer.py
class XLMRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = EncoderAttention(config)
        self.intermediate = EncoderIntermediate(config)
        self.output = EncoderOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)

    def forward_with_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attention.forward_with_attention_mask(
            hidden_states,
            attention_mask=attention_mask,
        )
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        attention_output = self.attention.forward_varlen(hidden_states, cu_seqlens, max_seqlen)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


# Inlined from tasks/baseline/L3/xlm_roberta_encoder.py
class XLMRobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([
            XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states

    def forward_with_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module.forward_with_attention_mask(
                hidden_states,
                attention_mask=attention_mask,
            )
        return hidden_states

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module.forward_varlen(hidden_states, cu_seqlens, max_seqlen)
        return hidden_states


# Inlined from tasks/baseline/L3/xlm_roberta_model.py
class XLMRobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        return attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors
        positions = positions + self.config.pad_token_id + 1
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
        )
        return self.encoder(embedding_output)

    def forward_with_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        position_ids = create_roberta_position_ids_from_input_ids(
            input_ids=input_ids,
            padding_idx=self.config.pad_token_id,
        )
        embedding_output = self.embeddings.forward_with_token_type_ids(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        extended_attention_mask = self._prepare_attention_mask(attention_mask, input_ids.device)
        return self.encoder.forward_with_attention_mask(
            embedding_output,
            attention_mask=extended_attention_mask,
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
        del intermediate_tensors
        positions = positions + self.config.pad_token_id + 1
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
        )
        return self.encoder.forward_varlen(embedding_output, cu_seqlens, max_seqlen)


from dataclasses import dataclass


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
        from transformers import AutoConfig
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
        hidden_states = hidden_states.to(self.colbert_linear.weight.dtype)
        vecs = self.colbert_linear(hidden_states[:, 1:])
        if attention_mask is not None:
            vecs = vecs * attention_mask[:, 1:].unsqueeze(-1).to(dtype=vecs.dtype)
        return self.norm(vecs)

    def token_classify(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.sparse_linear.weight.dtype)
        weights = self.relu(self.sparse_linear(hidden_states))
        if attention_mask is not None:
            weights = weights * attention_mask.unsqueeze(-1).to(dtype=weights.dtype)
        return weights
