from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fastkernels.tasks.baseline.L4.recurrent_cache import RecurrentCache, CausalLMOutputWithPast
from fastkernels.tasks.baseline.L4.gla import GLAConfig, GLAModel
from fastkernels.tasks.baseline.L1.linear import Linear


_CHUNK_THRESHOLD = 64


@triton.jit
def _rms_norm_k(X, W, Y, stride, N: tl.constexpr, eps: tl.constexpr, BLK: tl.constexpr):
    row = tl.program_id(0)
    off = row * stride
    c = tl.arange(0, BLK)
    m = c < N
    x = tl.load(X + off + c, mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x) / N + eps)
    w = tl.load(W + c, mask=m, other=1.0).to(tl.float32)
    tl.store(Y + off + c, (x * r * w).to(Y.dtype.element_ty), mask=m)


def _rms_norm(x, weight, eps):
    s = x.shape
    x2 = x.reshape(-1, s[-1])
    N = x2.shape[-1]
    y = torch.empty_like(x2)
    w = weight.to(x.dtype) if weight.dtype != x.dtype else weight
    _rms_norm_k[(x2.shape[0],)](x2, w, y, x2.stride(0), N, eps, triton.next_power_of_2(N))
    return y.view(s)


@triton.jit
def _logsig_div_k(X, Y, inv: tl.constexpr, stride, N: tl.constexpr, BLK: tl.constexpr):
    row = tl.program_id(0)
    off = row * stride
    c = tl.arange(0, BLK)
    m = c < N
    x = tl.load(X + off + c, mask=m, other=0.0).to(tl.float32)
    v = (tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))) * inv
    tl.store(Y + off + c, v.to(Y.dtype.element_ty), mask=m)


def _logsig_div(x, normalizer):
    s = x.shape
    x2 = x.reshape(-1, s[-1])
    N = x2.shape[-1]
    y = torch.empty_like(x2)
    _logsig_div_k[(x2.shape[0],)](x2, y, 1.0 / normalizer, x2.stride(0), N, triton.next_power_of_2(N))
    return y.view(s)


@triton.jit
def _norm_gate_k(O, G, W, Y, so, sg, hv: tl.constexpr, nH: tl.constexpr,
                  eps: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // nH
    hd = pid % nH
    bo = row * so + hd * hv
    bg = row * sg + hd * hv
    c = tl.arange(0, BLK)
    m = c < hv
    o = tl.load(O + bo + c, mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(o * o) / hv + eps)
    w = tl.load(W + c, mask=m, other=1.0).to(tl.float32)
    o = o * r * w
    g = tl.load(G + bg + c, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + bo + c, (o * g * tl.sigmoid(g)).to(Y.dtype.element_ty), mask=m)


def _norm_gate(o, g, w, nH, eps):
    M, D = o.shape
    hv = D // nH
    ww = w.to(o.dtype) if w.dtype != o.dtype else w
    y = torch.empty_like(o)
    _norm_gate_k[(M * nH,)](o, g, ww, y, o.stride(0), g.stride(0), hv, nH, eps,
                             triton.next_power_of_2(hv))
    return y


@triton.jit
def _silu_mul_k(X, Y, mid: tl.constexpr, stride: tl.constexpr, BLK: tl.constexpr):
    row = tl.program_id(0)
    b = row * stride
    c = tl.arange(0, BLK)
    m = c < mid
    a = tl.load(X + b + c, mask=m, other=0.0).to(tl.float32)
    u = tl.load(X + b + mid + c, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * mid + c, (a * tl.sigmoid(a) * u).to(Y.dtype.element_ty), mask=m)


def _silu_mul(gate_up):
    M, tot = gate_up.shape
    mid = tot // 2
    y = torch.empty(M, mid, dtype=gate_up.dtype, device=gate_up.device)
    _silu_mul_k[(M,)](gate_up, y, mid, gate_up.stride(0), triton.next_power_of_2(mid))
    return y


class GLAForCausalLM(nn.Module):
    def __init__(self, config: GLAConfig):
        super().__init__()
        self.config = config
        self.model = GLAModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.emb.weight
        self._D = config.hidden_size
        self._H = config.num_heads
        self._KD = int(config.hidden_size * config.expand_k)
        self._VD = int(config.hidden_size * config.expand_v)
        self._HK = self._KD // config.num_heads
        self._HV = self._VD // config.num_heads
        self._eps = config.norm_eps
        self._fused = False

    def load_state_dict(self, *a, **kw):
        self._fused = False
        return super().load_state_dict(*a, **kw)

    def _fuse(self):
        if self._fused:
            return
        for ly in self.model.layers:
            mp = ly.mlp
            mp.register_buffer(
                '_guw',
                torch.cat([mp.gate_proj.weight.data, mp.up_proj.weight.data], 0),
                persistent=False,
            )
        self._fused = True

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
        self._fuse()
        cu = kwargs.get("cu_seqlens")
        mx = None
        if cu is not None:
            lens = cu[1:] - cu[:-1]
            mx = int(lens.max().item()) if lens.numel() else 0
        if inputs_embeds is None:
            inputs_embeds = self.model.embeddings(input_ids)
        hs = inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = RecurrentCache()
        D, H, KD, VD = self._D, self._H, self._KD, self._VD
        HK, HV, eps = self._HK, self._HV, self._eps

        for ly in self.model.layers:
            B, T, _ = hs.shape
            att = ly.attn
            h = _rms_norm(hs.reshape(-1, D), ly.attn_norm.weight, eps)
            q = F.linear(h, att.q_proj.weight).view(B, T, H, HK)
            k = F.linear(h, att.k_proj.weight).view(B, T, H, HK)
            v = F.linear(h, att.v_proj.weight).view(B, T, H, HV)
            g = F.linear(h, att.g_proj.weight)
            gk = _logsig_div(
                F.linear(F.linear(h, att.gk_proj[0].weight),
                          att.gk_proj[1].weight, att.gk_proj[1].bias),
                att.gate_logit_normalizer,
            ).view(B, T, H, HK)
            s0 = None
            if past_key_values is not None and getattr(past_key_values, 'states', None):
                s0 = past_key_values.states.get(id(att))
            dl = mx if mx is not None else T
            if dl >= _CHUNK_THRESHOLD:
                o, sf = chunk_gla(q=q, k=k, v=v, g=gk, initial_state=s0,
                                   output_final_state=use_cache, cu_seqlens=cu)
            else:
                o, sf = fused_recurrent_gla(q=q, k=k, v=v, gk=gk, initial_state=s0,
                                             output_final_state=use_cache, cu_seqlens=cu)
            if use_cache and past_key_values is not None:
                if not hasattr(past_key_values, 'states'):
                    past_key_values.states = {}
                past_key_values.states[id(att)] = sf
            o2 = _norm_gate(o.reshape(-1, VD), g, att.g_norm_swish_gate.weight, H, eps)
            hs = hs + F.linear(o2, att.o_proj.weight).view(B, T, D)
            h = _rms_norm(hs.reshape(-1, D), ly.mlp_norm.weight, eps)
            hs = hs + F.linear(_silu_mul(F.linear(h, ly.mlp._guw)),
                                ly.mlp.down_proj.weight).view(B, T, D)

        nw = self.model.norm.weight
        hs2 = _rms_norm(hs.to(dtype=nw.dtype).reshape(-1, D), nw, eps).reshape_as(hs)
        if logits_indices is not None:
            hs2 = hs2.reshape(-1, D).index_select(0, logits_indices).unsqueeze(1)
        elif num_logits_to_keep > 0:
            hs2 = hs2[:, -num_logits_to_keep:, :]
        logits = F.linear(hs2, self.lm_head.weight).float()
        loss = None
        if labels is not None:
            sl = logits[..., :-1, :].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)),
                                    labels[..., 1:].contiguous().view(-1))
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, loss=loss)
