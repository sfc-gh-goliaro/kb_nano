"""Semantic PyTorch reference for variable-length Flash Attention."""


from __future__ import annotations


# Inlined helper (no baseline task): pure-torch dense/varlen attention
# and paged-cache gather.  The baselines call flash_attn / vllm_flash_attn
# here, so there is no baseline file to inline this from.
import torch
import torch.nn.functional as F


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
        # Materializing the full [.., q_len, k_len] score matrix in float32 costs
        # q_len*k_len*heads*4 bytes -- 64 GiB at 16384x16384 with 64 heads, which
        # is why gpt_oss_decoder's tokens-16384 scenarios died with an OOM while
        # the FlashAttention baseline peaked near 13 GB.  Walk the query axis in
        # blocks so peak memory is O(block * k_len).  Softmax runs along the key
        # axis, so every block is a complete softmax over all keys and the result
        # is identical, not an approximation.
        q_block = 1024 if q_len * k_len > (8192 * 8192) else q_len
        out_parts = []
        for start in range(0, q_len, q_block):
            stop = min(start + q_block, q_len)
            q_chunk = q_in[..., start:stop, :]
            mask_chunk = mask[start:stop]
            sc = torch.matmul(
                q_chunk.float(), k_in.float().transpose(-2, -1)
            ) * scale
            if softcap > 0.0:
                sc = torch.tanh(sc / softcap) * softcap
            sc = sc.masked_fill(~mask_chunk, torch.finfo(sc.dtype).min)
            if s_aux is not None:
                sink = s_aux.to(device=sc.device, dtype=sc.dtype).view(1, -1, 1, 1)
                sink = sink.expand(sc.shape[0], -1, sc.shape[-2], -1)
                p = torch.softmax(torch.cat((sc, sink), dim=-1), dim=-1)[..., :-1]
            else:
                p = torch.softmax(sc, dim=-1)
            p = p.masked_fill(torch.all(~mask_chunk, dim=-1, keepdim=True), 0.0)
            if s_aux is not None:
                out_parts.append(torch.matmul(p, v_in.float()).to(v_in.dtype))
            else:
                out_parts.append(torch.matmul(p.to(v_in.dtype), v_in))
            del sc, p
        out = torch.cat(out_parts, dim=-2) if len(out_parts) > 1 else out_parts[0]
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


import torch.nn as nn


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
