from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

try:
    from fastkernels.tasks.baseline.L1.chunk_gla import ChunkGLA
    from fastkernels.tasks.baseline.L1.chunk_retention import ChunkRetention
    from fastkernels.tasks.baseline.L1.fused_recurrent_gla import FusedRecurrentGLA
    from fastkernels.tasks.baseline.L1.fused_recurrent_retention import FusedRecurrentRetention
    from fastkernels.tasks.baseline.L1.gla_recurrence import NaiveRecurrentGLA
    from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
    from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
    from fastkernels.tasks.baseline.L4.recurrent_cache import CausalLMOutputWithPast, RecurrentCache
except ModuleNotFoundError:
    sys.path.insert(0, "/mnt/weka/home/hao.zhang/async_rl_bench")
    from kb_nano.tasks.baseline.L1.chunk_gla import ChunkGLA
    from kb_nano.tasks.baseline.L1.chunk_retention import ChunkRetention
    from kb_nano.tasks.baseline.L1.fused_recurrent_gla import FusedRecurrentGLA
    from kb_nano.tasks.baseline.L1.fused_recurrent_retention import FusedRecurrentRetention
    from kb_nano.tasks.baseline.L1.gla_recurrence import NaiveRecurrentGLA
    from kb_nano.tasks.baseline.L1.rms_norm import RMSNorm
    from kb_nano.tasks.baseline.L1.rotary_emb import RotaryEmbedding
    from kb_nano.tasks.baseline.L4.recurrent_cache import CausalLMOutputWithPast, RecurrentCache


_CHUNK_THRESHOLD = 64


if _TRITON_AVAILABLE:

    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=0)
        inv = tl.rsqrt(ss / n_cols + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + row * n_cols + offs, x * inv * w, mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        a_ptr,
        b_ptr,
        w_ptr,
        out_ptr,
        y_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        idx = row * n_cols + offs
        a = tl.load(a_ptr + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
        v = (a + b).to(tl.float32)
        ss = tl.sum(v * v, axis=0)
        inv = tl.rsqrt(ss / n_cols + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + idx, v, mask=mask)
        tl.store(y_ptr + idx, v * inv * w, mask=mask)


def _num_warps(block: int) -> int:
    if block >= 4096:
        return 8
    if block >= 2048:
        return 4
    return 1


def _can_use_triton(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and weight.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.ndim >= 2
        and x.shape[-1] == weight.numel()
        and x.shape[-1] <= 65536
        and x.numel() > 0
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _fallback_rms_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return norm(x.reshape(-1, x.size(-1))).reshape_as(x)


def _fast_rms_norm(norm: nn.Module, x: torch.Tensor, eps: float) -> torch.Tensor:
    weight = norm.weight
    if not _can_use_triton(x, weight):
        return _fallback_rms_norm(norm, x)
    n_cols = x.shape[-1]
    rows = x.numel() // n_cols
    y = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    _rms_norm_kernel[(rows,)](
        x, weight, y, n_cols, eps, BLOCK=block, num_warps=_num_warps(block)
    )
    return y


def _fast_add_rms_norm(
    norm: nn.Module,
    residual: torch.Tensor,
    update: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = norm.weight
    if not (
        _can_use_triton(residual, weight)
        and update.is_cuda
        and update.is_contiguous()
        and update.shape == residual.shape
        and update.dtype == residual.dtype
    ):
        out = residual + update
        return out, _fallback_rms_norm(norm, out)
    n_cols = residual.shape[-1]
    rows = residual.numel() // n_cols
    out = torch.empty_like(residual)
    y = torch.empty_like(residual)
    block = triton.next_power_of_2(n_cols)
    _add_rms_norm_kernel[(rows,)](
        residual, update, weight, out, y, n_cols, eps,
        BLOCK=block, num_warps=_num_warps(block)
    )
    return out, y


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        return self.emb(input_ids)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            F.silu(F.linear(x, self.gate_proj.weight, None))
            * F.linear(x, self.up_proj.weight, None),
            self.down_proj.weight,
            None,
        )


class GatedLinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        decay_mode: str = "learned_low_rank",
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        use_rotary: bool = False,
        rotary_base: float = 10000.0,
        rotary_max_position: int = 8192,
        norm_eps: float = 1e-6,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.decay_mode = decay_mode
        self.use_rotary = use_rotary
        self.gate_logit_normalizer = gate_logit_normalizer
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.q_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)
        if decay_mode == "learned_low_rank":
            self.gk_proj = nn.Sequential(
                Linear(hidden_size, gate_low_rank_dim, bias=False),
                Linear(gate_low_rank_dim, self.key_dim, bias=True),
            )
        else:
            h_idx = torch.arange(num_heads, dtype=torch.float32)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32), -5.0 - h_idx)
            self.register_buffer("log_gamma", torch.log(gamma), persistent=False)
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_k_dim,
                max_position_embeddings=rotary_max_position,
                rope_theta=rotary_base,
            )
        self.use_fast_kernels = use_fast_kernels
        self.naive_recurrence = NaiveRecurrentGLA()
        if use_fast_kernels:
            if decay_mode == "learned_low_rank":
                self.fused_recurrence = FusedRecurrentGLA()
                self.chunk = ChunkGLA()
            else:
                self.fused_recurrence = FusedRecurrentRetention()
                self.chunk = ChunkRetention()
        self.g_norm_swish_gate = RMSNorm(self.head_v_dim, eps=norm_eps)
        self._norm_eps = float(norm_eps)

    def _compute_gk(self, hidden_states: torch.Tensor, B: int, T: int) -> torch.Tensor:
        if self.decay_mode == "learned_low_rank":
            gk = F.linear(
                F.linear(hidden_states, self.gk_proj[0].weight, None),
                self.gk_proj[1].weight,
                self.gk_proj[1].bias,
            )
            gk = F.logsigmoid(gk) / self.gate_logit_normalizer
            return gk.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)
        return self.log_gamma.to(hidden_states.dtype).view(
            1, self.num_heads, 1, 1
        ).expand(B, self.num_heads, T, self.head_k_dim)

    def _compute_gk_bthk(self, hidden_states: torch.Tensor, B: int, T: int) -> torch.Tensor:
        gk = F.linear(
            F.linear(hidden_states, self.gk_proj[0].weight, None),
            self.gk_proj[1].weight,
            self.gk_proj[1].bias,
        )
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        return gk.view(B, T, self.num_heads, self.head_k_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        B, T, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        max_seqlen = None
        if cu_seqlens is not None:
            if B != 1:
                raise ValueError("cu_seqlens prefill expects packed hidden_states with batch size 1")
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = int(lengths.max().item()) if lengths.numel() else 0

        q = F.linear(hidden_states, self.q_proj.weight, None)
        k = F.linear(hidden_states, self.k_proj.weight, None)
        v = F.linear(hidden_states, self.v_proj.weight, None)
        g = F.linear(hidden_states, self.g_proj.weight, None)

        if self.use_rotary:
            offsets = getattr(past_key_values, "seq_offsets", None) if past_key_values is not None else None
            local = torch.arange(T, device=q.device, dtype=torch.int64)
            if cu_seqlens is not None:
                seg_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                seg_starts = cu_seqlens[:-1].to(torch.int64)
                local_within = local - torch.repeat_interleave(seg_starts, seg_lengths)
                if offsets is None:
                    positions = local_within.contiguous()
                else:
                    off_t = (
                        offsets.to(device=q.device, dtype=torch.int64)
                        if torch.is_tensor(offsets)
                        else torch.full((seg_lengths.numel(),), int(offsets), device=q.device, dtype=torch.int64)
                    )
                    positions = (torch.repeat_interleave(off_t, seg_lengths) + local_within).contiguous()
            elif offsets is None:
                positions = local.repeat(B)
            elif isinstance(offsets, int):
                positions = (local + offsets).repeat(B)
            else:
                positions = (
                    offsets.to(device=q.device, dtype=torch.int64).unsqueeze(1)
                    + local.unsqueeze(0)
                ).reshape(-1).contiguous()
            q_flat = q.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            k_flat = k.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(B, T, self.num_heads, self.head_k_dim)
            k = k_flat.view(B, T, self.num_heads, self.head_k_dim)
        else:
            q = q.view(B, T, self.num_heads, self.head_k_dim)
            k = k.view(B, T, self.num_heads, self.head_k_dim)
        v = v.view(B, T, self.num_heads, self.head_v_dim)

        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))

        if self.use_fast_kernels and q.is_cuda:
            dispatch_len = max_seqlen if max_seqlen is not None else T
            if self.decay_mode == "learned_low_rank":
                gk = self._compute_gk_bthk(hidden_states, B, T)
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v, g=gk, initial_state=initial_state,
                        output_final_state=use_cache, cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v, gk=gk, initial_state=initial_state,
                        output_final_state=use_cache, cu_seqlens=cu_seqlens,
                    )
            elif dispatch_len >= _CHUNK_THRESHOLD:
                o, final_state = self.chunk(
                    q=q, k=k, v=v, initial_state=initial_state,
                    output_final_state=use_cache, cu_seqlens=cu_seqlens,
                )
            else:
                o, final_state = self.fused_recurrence(
                    q=q, k=k, v=v, initial_state=initial_state,
                    output_final_state=use_cache, cu_seqlens=cu_seqlens,
                )
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            gk = self._compute_gk(hidden_states, B, T)
            o, final_state = self.naive_recurrence(
                q, k, v, gk, initial_state=initial_state, output_final_state=use_cache
            )
            o = o.transpose(1, 2)

        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state

        o = _fast_rms_norm(self.g_norm_swish_gate, o.reshape(-1, self.head_v_dim), self._norm_eps)
        o = o.view(B, T, self.value_dim)
        o = o * F.silu(g)
        return F.linear(o, self.o_proj.weight, None), None, past_key_values


class GLADecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedLinearAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            decay_mode=getattr(config, "decay_mode", "learned_low_rank"),
            gate_low_rank_dim=getattr(config, "gate_low_rank_dim", 16),
            gate_logit_normalizer=getattr(config, "gate_logit_normalizer", 16),
            use_rotary=getattr(config, "use_rotary", False),
            rotary_base=getattr(config, "rotary_base", 10000.0),
            rotary_max_position=getattr(config, "max_position_embeddings", 8192),
            norm_eps=config.norm_eps,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(config.hidden_size, config.intermediate_size)
        self._norm_eps = float(config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        residual = hidden_states
        h = _fast_rms_norm(self.attn_norm, hidden_states, self._norm_eps)
        h, attentions, past_key_values = self.attn(
            hidden_states=h,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states, h = _fast_add_rms_norm(self.mlp_norm, residual, h, self._norm_eps)
        h = self.mlp(h)
        h.add_(hidden_states)
        return h, attentions, past_key_values


@dataclass
class GLAConfig:
    hidden_size: int = 2560
    num_heads: int = 5
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    expand_k: float = 0.5
    expand_v: float = 1.0
    hidden_ratio: int = 4
    intermediate_size: int | None = None
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.intermediate_size is None:
            intermediate = int(self.hidden_size * self.hidden_ratio * 2 / 3)
            self.intermediate_size = 256 * ((intermediate + 255) // 256)

    @classmethod
    def from_dict(cls, data: dict) -> "GLAConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values() if f.name != "dtype"}
        kwargs = {k: data[k] for k in keys if k in data}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "GLAConfig":
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class GLAModel(nn.Module):
    def __init__(self, config: GLAConfig):
        super().__init__()
        self.embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GLADecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self._norm_eps = float(config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: RecurrentCache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, RecurrentCache | None]:
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = RecurrentCache()
        for layer in self.layers:
            hidden_states, _, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        norm_dtype = self.norm.weight.dtype
        h = hidden_states.to(dtype=norm_dtype)
        hidden_states = _fast_rms_norm(self.norm, h, self._norm_eps).reshape_as(hidden_states)
        return hidden_states, past_key_values


class GLAForCausalLM(nn.Module):
    def __init__(self, config: GLAConfig):
        super().__init__()
        self.config = config
        self.model = GLAModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.emb.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: RecurrentCache | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        num_logits_to_keep: int = 0,
        logits_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        if logits_indices is not None:
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            hidden_states = hidden_states.index_select(0, logits_indices).unsqueeze(1)
        elif num_logits_to_keep > 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        logits = F.linear(hidden_states, self.lm_head.weight, self.lm_head.bias).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithPast(
            logits=logits, past_key_values=past_key_values, loss=loss,
        )
