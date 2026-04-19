"""EAGLE-3 speculative-decoding engine for kb_nano (eager, tree draft).

Implements EAGLE-3 with K-branch tree drafting (matches sglang's
``select_top_k_tokens`` + ``build_tree_kernel_efficient`` + ``verify_tree_greedy``
pipeline). The default config mirrors sglang's reference defaults:
``spec_steps=3``, ``topk=4``, ``num_draft_tokens=16``.

For each speculative step:

  1. ``_draft_chain`` runs S-1 forward passes on the draft model, producing K
     candidates per seq per step. Each step expands K * K candidates and selects
     the top K by cumulative log-probability. All branches share the prefix
     KV (set up by the most recent ``_draft_extend``); each branch additionally
     writes to a per-branch tail block in the draft paged KV cache.
  2. ``build_tree_kernel_efficient`` packs the K * (S-1) + K candidates into a
     flat tree of ``num_draft_tokens`` verify positions with the right
     parent / sibling / position table.
  3. ``_target_verify`` runs the target on the flat tree and ``verify_tree_greedy``
     selects the longest accepted path.
  4. ``_remap_target_kv_after_verify`` re-orders target KV slots so logical
     position ``t_committed_len + k`` holds the K/V for the k-th accepted
     token (vs. the k-th tree node in flat write order). This is a no-op
     for chain drafting but required for correctness with tree drafting.
  5. ``_draft_extend`` digests the accepted tokens (with target aux hidden
     states) into the draft KV cache, capturing the next round's K topk.

Scope:
  - single GPU, TP=1
  - eager forwards (no CUDA graphs) on both target and draft
  - greedy verify only (T=0)
  - one batch in flight at a time
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from .context import get_attn_backend_config, set_context, set_forward_context
from .weight_loader import load_eagle3_draft_model, load_model
from ..tasks.baseline.L1.eagle_tree_ops import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    organize_draft_results,
    verify_tree_greedy,
)


@dataclass
class Eagle3SamplingParams:
    max_tokens: int = 512
    ignore_eos: bool = False


@dataclass
class Eagle3Output:
    prompt_token_ids: List[int]
    token_ids: List[int]
    generated_text: str = ""


class _PagedKVCache:
    """Lightweight paged KV cache + free list for either target or draft."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        layout: str = "NHD",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.layout = layout

        if layout == "HND":
            shape = (2, num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        else:
            shape = (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.kv = torch.empty(*shape, dtype=dtype, device=device)
        self._free = list(range(num_blocks))

    def reset(self):
        self._free = list(range(self.num_blocks))

    def num_free_blocks(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> List[int]:
        assert n <= len(self._free), f"OOM: need {n} blocks, have {len(self._free)}"
        out = self._free[:n]
        self._free = self._free[n:]
        return out

    def free(self, blocks: List[int]):
        self._free.extend(blocks)

    def attach(self, attn_layers):
        assert len(attn_layers) == self.num_layers, (
            f"layer count mismatch: cache has {self.num_layers} layers but "
            f"model has {len(attn_layers)} attn layers"
        )
        for i, mod in enumerate(attn_layers):
            mod.k_cache = self.kv[0, i]
            mod.v_cache = self.kv[1, i]


class _Eagle3Sequence:
    _next_id = 0

    def __init__(self, prompt_ids: List[int], max_tokens: int, ignore_eos: bool):
        _Eagle3Sequence._next_id += 1
        self.seq_id = _Eagle3Sequence._next_id
        self.prompt_ids = list(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.generated_ids: List[int] = []
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos

        self.t_blocks: List[int] = []
        self.d_blocks: List[int] = []
        # Branch-specific tail blocks for the current draft chain (allocated
        # at the start of each ``_draft_chain`` and freed before the next
        # draft extend). Length K per seq.
        self._chain_tail_blocks: List[int] = []

        # Draft hidden state (target hidden3-cat) at the last accepted token.
        self.last_aux: Optional[torch.Tensor] = None
        # The last accepted token id.
        self.last_token: Optional[int] = None
        # Number of tokens whose KV is committed in the target paged cache.
        self.t_committed_len: int = 0
        # Number of positions populated in the draft KV cache.
        # In sglang convention this stays equal to ``t_committed_len`` after
        # each draft-extend (once the draft has digested the latest accepted
        # tokens). Between the chain forwards and the post-verify extend it
        # may transiently exceed t_committed_len; we always free those extra
        # slots before the next extend.
        self.d_committed_len: int = 0

        # Output of the latest draft-extend: the top-K logprobs / token ids
        # (in draft vocab) for the FIRST speculative position, and the draft
        # hidden state at the last extend position. These feed chain step 0.
        self.draft_topk_p: Optional[torch.Tensor] = None       # [K] (float)
        self.draft_topk_i: Optional[torch.Tensor] = None       # [K] (long, draft vocab)
        self.draft_topk_i_t: Optional[torch.Tensor] = None     # [K] (long, target vocab)
        self.draft_hidden: Optional[torch.Tensor] = None       # [H_d]

        self.finished = False


def _gather_attn_layers(model) -> list:
    out = []
    for m in model.modules():
        if hasattr(m, "k_cache") and hasattr(m, "v_cache") and hasattr(m, "scale"):
            out.append(m)
    return out


class LlamaEagle3Engine:
    """Single-process EAGLE-3 engine with chain drafting (eager, TP=1)."""

    def __init__(
        self,
        model_name: str,
        draft_repo: str,
        seed: int = 42,
        dtype: torch.dtype = torch.bfloat16,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        max_num_seqs: int = 32,
        spec_steps: int = 3,
        spec_topk: int = 4,
        num_draft_tokens: Optional[int] = None,
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda")
        self.dtype = dtype
        self.spec_steps = spec_steps
        self.topk = spec_topk
        # sglang default: 1 (root) + topk + (spec_steps - 1) * topk * topk
        # candidates produced; the build_tree kernel keeps the top
        # num_draft_tokens - 1 by score and prepends the root.
        if num_draft_tokens is None:
            num_draft_tokens = 1 + spec_topk + (spec_steps - 1) * spec_topk * spec_topk
        self.num_draft_tokens = num_draft_tokens
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault(
                "MASTER_PORT", os.environ.get("KB_NANO_EAGLE3_PORT", "29503"),
            )
            torch.cuda.set_device(0)
            dist.init_process_group(
                "nccl", rank=0, world_size=1,
                device_id=torch.device("cuda", 0),
            )

        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")

        attn_cfg = get_attn_backend_config()
        self.block_size = attn_cfg.block_size
        self.kv_layout = attn_cfg.kv_layout

        print(f"[EAGLE-3] Loading target model: {model_name}")
        self.target, self.target_config = load_model(
            model_name, device=self.device, dtype=dtype,
        )
        self.target.set_eagle3_layers_to_capture(None)

        print(f"[EAGLE-3] Loading draft model: {draft_repo}")
        self.draft, self.draft_config = load_eagle3_draft_model(
            draft_repo, self.target_config, device=self.device, dtype=dtype,
        )
        self.draft.set_embed_tokens(self.target.model.embed_tokens)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.target_attn_layers = _gather_attn_layers(self.target)
        self.draft_attn_layers = _gather_attn_layers(self.draft)

        self._allocate_kv_caches(gpu_memory_utilization)

        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def _allocate_kv_caches(self, gpu_mem_util: float):
        free, total = torch.cuda.mem_get_info()
        used = total - free

        elem_size = torch.finfo(self.dtype).bits // 8
        n_t_layers = len(self.target_attn_layers)
        n_d_layers = len(self.draft_attn_layers)
        n_kv_t = self.target_config.num_key_value_heads
        n_kv_d = self.draft_config.num_key_value_heads
        head_dim_t = self.target_config.head_dim
        head_dim_d = self.draft_config.head_dim

        block_bytes_t = (
            2 * n_t_layers * self.block_size * n_kv_t * head_dim_t * elem_size
        )
        block_bytes_d = (
            2 * n_d_layers * self.block_size * n_kv_d * head_dim_d * elem_size
        )

        available = max(0, int(total * gpu_mem_util) - used)
        # Reserve some headroom for activations / draft tree buffers (~512 MiB).
        available = max(0, available - (512 << 20))

        target_budget = int(available * 0.97)
        draft_budget = available - target_budget

        n_blocks_t = max(64, target_budget // block_bytes_t)
        n_blocks_d = max(64, draft_budget // block_bytes_d)

        max_blocks_per_seq = (
            self.max_model_len + self.block_size - 1
        ) // self.block_size
        n_blocks_t = min(n_blocks_t, self.max_num_seqs * max_blocks_per_seq + 64)
        n_blocks_d = min(n_blocks_d, self.max_num_seqs * max_blocks_per_seq + 64)

        print(f"[EAGLE-3] Target KV cache: {n_blocks_t} blocks "
              f"x {self.block_size} = {n_blocks_t * self.block_size} slots")
        print(f"[EAGLE-3] Draft  KV cache: {n_blocks_d} blocks "
              f"x {self.block_size} = {n_blocks_d * self.block_size} slots")

        self.target_kv = _PagedKVCache(
            n_t_layers, n_kv_t, head_dim_t, n_blocks_t, self.block_size,
            layout=self.kv_layout, dtype=self.dtype, device=self.device,
        )
        self.draft_kv = _PagedKVCache(
            n_d_layers, n_kv_d, head_dim_d, n_blocks_d, self.block_size,
            layout=self.kv_layout, dtype=self.dtype, device=self.device,
        )
        self.target_kv.attach(self.target_attn_layers)
        self.draft_kv.attach(self.draft_attn_layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_target_capacity(self, seq: _Eagle3Sequence, target_len: int):
        needed = (target_len + self.block_size - 1) // self.block_size
        cur = len(seq.t_blocks)
        if needed > cur:
            seq.t_blocks.extend(self.target_kv.alloc(needed - cur))

    def _ensure_draft_capacity(self, seq: _Eagle3Sequence, target_len: int):
        needed = (target_len + self.block_size - 1) // self.block_size
        cur = len(seq.d_blocks)
        if needed > cur:
            seq.d_blocks.extend(self.draft_kv.alloc(needed - cur))

    def _free_draft_tail(self, seq: _Eagle3Sequence, keep_len: int):
        """Free draft KV blocks beyond ``keep_len`` slots."""
        keep_blocks = (keep_len + self.block_size - 1) // self.block_size
        if keep_blocks < len(seq.d_blocks):
            released = seq.d_blocks[keep_blocks:]
            self.draft_kv.free(released)
            seq.d_blocks = seq.d_blocks[:keep_blocks]

    def _slot_mapping_for_range(
        self, blocks: List[int], start: int, end: int,
    ) -> List[int]:
        out = []
        for p in range(start, end):
            blk = blocks[p // self.block_size]
            out.append(blk * self.block_size + (p % self.block_size))
        return out

    def _slot_for(self, blocks: List[int], pos: int) -> int:
        return blocks[pos // self.block_size] * self.block_size + (pos % self.block_size)

    @torch.inference_mode()
    def _remap_target_kv_after_verify(
        self,
        seqs: List[_Eagle3Sequence],
        accept_index_cpu: List[List[int]],
        accept_num_cpu: List[int],
    ):
        """After tree verify, K/V for tree node ``i`` was written at the slot
        for logical position ``t_committed_len + i`` (linear write order).
        For the next iteration we need K/V for the k-th accepted token at the
        slot for logical position ``t_committed_len + k``. With chain drafting
        accepted tree indices are already ``[0, 1, 2, ...]`` so this is a no-op,
        but with tree drafting the accepted chain typically picks non-contiguous
        tree nodes and we MUST remap, otherwise subsequent attention reads
        garbage K/V from rejected branches.

        Mirrors sglang's ``move_kv_cache(tgt_cache_loc, src_cache_loc)``.
        """
        N = self.num_draft_tokens
        src_slots: list[int] = []
        dst_slots: list[int] = []
        for i, seq in enumerate(seqs):
            n_accept = int(accept_num_cpu[i])
            base = seq.t_committed_len
            for k in range(n_accept + 1):
                flat = int(accept_index_cpu[i][k])
                tree_idx = flat - i * N
                if tree_idx == k:
                    continue
                src_slots.append(self._slot_for(seq.t_blocks, base + tree_idx))
                dst_slots.append(self._slot_for(seq.t_blocks, base + k))

        if not src_slots:
            return

        src_t = torch.tensor(src_slots, device=self.device, dtype=torch.long)
        dst_t = torch.tensor(dst_slots, device=self.device, dtype=torch.long)
        kv = self.target_kv.kv
        BS = self.block_size
        if self.kv_layout == "NHD":
            # kv: [2, L, NB, BS, H, D] -> view as [2, L, NB*BS, H, D].
            flat = kv.view(
                2, self.target_kv.num_layers,
                self.target_kv.num_blocks * BS,
                self.target_kv.num_kv_heads, self.target_kv.head_dim,
            )
            src_vals = flat[:, :, src_t].clone()
            flat[:, :, dst_t] = src_vals
        else:
            # HND: kv [2, L, NB, H, BS, D]. NB and BS are non-adjacent so
            # we cannot trivially view as [..., NB*BS, ...]. Instead index
            # block / offset directly per slot.
            src_blk = (src_t // BS).tolist()
            src_off = (src_t % BS).tolist()
            dst_blk = (dst_t // BS).tolist()
            dst_off = (dst_t % BS).tolist()
            for n in range(len(src_blk)):
                kv[:, :, dst_blk[n], :, dst_off[n], :] = \
                    kv[:, :, src_blk[n], :, src_off[n], :].clone()

    # ------------------------------------------------------------------
    # Target prefill on the prompt
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _target_prefill(self, seqs: List[_Eagle3Sequence]):
        """Run target prefill on the prompts.

        Returns ``aux_per_seq``: list of per-seq aux-cat tensors of shape
        ``[P_i, 3*H_t]`` (concat of the three captured target hidden states).
        These feed the initial draft-extend.
        """
        input_ids: list[int] = []
        positions: list[int] = []
        cu_q = [0]
        cu_k = [0]
        slot_mapping: list[int] = []
        max_blocks = 0

        for seq in seqs:
            sl = len(seq.prompt_ids)
            self._ensure_target_capacity(seq, sl)
            input_ids.extend(seq.prompt_ids)
            positions.extend(range(sl))
            cu_q.append(cu_q[-1] + sl)
            cu_k.append(cu_k[-1] + sl)
            slot_mapping.extend(self._slot_mapping_for_range(seq.t_blocks, 0, sl))
            max_blocks = max(max_blocks, len(seq.t_blocks))
            seq.t_committed_len = sl

        n = len(seqs)
        bt = np.full((n, max_blocks), -1, dtype=np.int32)
        for i, seq in enumerate(seqs):
            bt[i, :len(seq.t_blocks)] = seq.t_blocks

        max_sq = max(len(s.prompt_ids) for s in seqs)
        max_sk = max_sq

        set_context(
            True,
            torch.tensor(cu_q, dtype=torch.int32, device=self.device),
            torch.tensor(cu_k, dtype=torch.int32, device=self.device),
            max_sq, max_sk,
            torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            block_tables=torch.from_numpy(bt).to(self.device),
        )
        ids_t = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        pos_t = torch.tensor(positions, dtype=torch.int64, device=self.device)

        out = self.target.model(ids_t, pos_t)
        if isinstance(out, tuple):
            hidden_states, aux_list = out
        else:
            hidden_states, aux_list = out, []

        # compute_logits internally slices to last-token-per-seq using context.
        logits = self.target.compute_logits(hidden_states)
        next_ids = logits.argmax(dim=-1).tolist()

        # Collect full aux per seq (for the initial draft extend).
        aux_per_seq: list[torch.Tensor] = []
        for i in range(n):
            s_idx, e_idx = cu_q[i], cu_q[i + 1]
            if len(aux_list) > 0:
                aux_per_seq.append(
                    torch.cat([a[s_idx:e_idx] for a in aux_list], dim=-1)
                )
            else:
                aux_per_seq.append(torch.zeros(
                    (e_idx - s_idx, self.target_config.hidden_size * 3),
                    device=self.device, dtype=self.dtype,
                ))

        for i, seq in enumerate(seqs):
            seq.last_token = int(next_ids[i])
            seq.last_aux = aux_per_seq[i][-1].clone()
            seq.token_ids.append(seq.last_token)
            seq.generated_ids.append(seq.last_token)

        return aux_per_seq

    # ------------------------------------------------------------------
    # Draft EXTEND: digest a stretch of tokens (with target aux) into draft KV
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _draft_extend(
        self,
        seqs: List[_Eagle3Sequence],
        ext_input_ids: list[torch.Tensor],   # per-seq, shape [L_i] target-vocab ids
        ext_hiddens: list[torch.Tensor],     # per-seq, shape [L_i, 3*H_t]
        ext_lens: list[int],                 # per-seq L_i
    ):
        """Run the draft model in extend mode and capture, per-seq, the
        first-step (topk_p, topk_i, topk_i_t, hidden) for the next chain.

        Each seq's draft KV is grown from ``d_committed_len`` to
        ``d_committed_len + L_i``. Updates ``d_committed_len`` in-place.
        """
        bs = len(seqs)
        K = self.topk

        positions: list[int] = []
        cu_q = [0]
        cu_k = [0]
        slot_mapping: list[int] = []
        max_blocks = 0
        last_idx_per_seq: list[int] = []

        for i, seq in enumerate(seqs):
            L = ext_lens[i]
            prefix = seq.d_committed_len
            new_total = prefix + L
            self._ensure_draft_capacity(seq, new_total)
            positions.extend(range(prefix, new_total))
            cu_q.append(cu_q[-1] + L)
            cu_k.append(cu_k[-1] + new_total)
            slot_mapping.extend(
                self._slot_mapping_for_range(seq.d_blocks, prefix, new_total)
            )
            max_blocks = max(max_blocks, len(seq.d_blocks))
            last_idx_per_seq.append(cu_q[-1] - 1)

        bt = np.full((bs, max_blocks), -1, dtype=np.int32)
        for i, seq in enumerate(seqs):
            bt[i, :len(seq.d_blocks)] = seq.d_blocks

        max_sq = max(ext_lens)
        max_sk = max(seqs[i].d_committed_len + ext_lens[i] for i in range(bs))

        set_context(
            True,
            torch.tensor(cu_q, dtype=torch.int32, device=self.device),
            torch.tensor(cu_k, dtype=torch.int32, device=self.device),
            max_sq, max_sk,
            torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            block_tables=torch.from_numpy(bt).to(self.device),
        )

        ids_t = torch.cat(ext_input_ids).to(torch.int64).to(self.device)
        pos_t = torch.tensor(positions, dtype=torch.long, device=self.device)
        hidden_t = torch.cat(ext_hiddens, dim=0).to(self.device)

        if os.environ.get("KB_NANO_EAGLE3_DEBUG") == "1":
            print(f"  [_draft_extend] ids_t shape={ids_t.shape} dtype={ids_t.dtype} "
                  f"min={int(ids_t.min())} max={int(ids_t.max())}", flush=True)
            print(f"  [_draft_extend] pos_t shape={pos_t.shape} max={int(pos_t.max())}", flush=True)
            print(f"  [_draft_extend] hidden_t shape={hidden_t.shape} dtype={hidden_t.dtype}", flush=True)
            print(f"  [_draft_extend] last_idx_per_seq={last_idx_per_seq}", flush=True)

        hidden_to_logits, hidden_to_aux = self.draft.forward_draft(
            ids_t, pos_t, hidden_t,
        )

        if os.environ.get("KB_NANO_EAGLE3_DEBUG") == "1":
            print(f"  [_draft_extend] hidden_to_logits shape={hidden_to_logits.shape}", flush=True)

        # ``compute_logits`` (ParallelLMHead.project) slices to the last
        # token per seq using cu_seqlens_q when ctx.is_prefill=True, so we
        # pass the FULL hidden_to_logits tensor.
        logits = self.draft.compute_logits(hidden_to_logits)       # [bs, draft_vocab]

        last_idx_t = torch.tensor(
            last_idx_per_seq, device=self.device, dtype=torch.long,
        )
        last_aux_h = hidden_to_aux[last_idx_t]                     # [bs, H_d]

        log_probs = torch.log_softmax(logits, dim=-1)
        top_p, top_i = torch.topk(log_probs, K, dim=-1)            # [bs, K]
        top_i_target = self.draft.remap_draft_ids(top_i)           # [bs, K]

        for i, seq in enumerate(seqs):
            seq.d_committed_len += ext_lens[i]
            seq.draft_topk_p = top_p[i].clone()
            seq.draft_topk_i = top_i[i].clone()
            seq.draft_topk_i_t = top_i_target[i].clone()
            seq.draft_hidden = last_aux_h[i].clone()

    # ------------------------------------------------------------------
    # Draft chain (topk=K, S steps). Tree drafting:
    #   - Step 0: record the K candidates from the latest draft-extend.
    #   - Forward step f (f=0..S-2): run B*K decode queries, each branch in
    #     its own per-branch tail block; produce K new candidates per branch
    #     and reduce K*K -> K by cumulative log-prob (sglang select_top_k).
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _draft_chain(self, seqs: List[_Eagle3Sequence]):
        """Produce sglang-format tree inputs from S draft steps with K branches."""
        S = self.spec_steps
        K = self.topk
        bs = len(seqs)

        verified_id = torch.tensor(
            [s.last_token for s in seqs], device=self.device, dtype=torch.long,
        )

        # Pull initial state from most recent draft extend.
        cur_top_p = torch.stack([s.draft_topk_p for s in seqs])           # [bs, K]
        cur_top_i_t = torch.stack([s.draft_topk_i_t for s in seqs])       # [bs, K]
        cur_hidden = torch.stack([s.draft_hidden for s in seqs])          # [bs, H_d]

        score_list: list[torch.Tensor] = []
        token_list: list[torch.Tensor] = []
        parents_list: list[torch.Tensor] = []

        # Step 0: just record (no forward).
        score_list.append(cur_top_p.unsqueeze(1))                          # [bs, 1, K]
        token_list.append(cur_top_i_t.view(bs, K))                         # [bs, K]
        parents_list.append(
            torch.arange(-1, K, dtype=torch.long, device=self.device)
            .unsqueeze(0).repeat(bs, 1)                                    # [bs, K + 1]
        )

        if S <= 1:
            parent_list, top_scores_index, draft_tokens = organize_draft_results(
                score_list, token_list, parents_list, self.num_draft_tokens,
            )
            return parent_list, top_scores_index, draft_tokens, verified_id

        # Per-(seq, branch) initial state for the forward loop.
        input_ids_bk = cur_top_i_t.reshape(bs * K).to(torch.int64)
        hidden_bk = cur_hidden.repeat_interleave(K, dim=0)                # [bs*K, H_d]
        scores = cur_top_p                                                # [bs, K]

        # Build per-branch block tables. Each branch needs an ISOLATED tail
        # past d_committed_len (so K branches don't collide on shared slots).
        # When d_committed_len isn't block-aligned, the last prefix block has
        # `r = d_committed_len % block_size` real slots and `block_size - r`
        # unused-but-shared slots. The branch tail starts at position
        # d_committed_len, which lies in the partial last prefix block. We
        # therefore COPY the partial last prefix block per branch and append
        # extra blocks if the tail spills past one block_size boundary.
        bsz = self.block_size
        max_tail_pos = (S - 2)  # last forward writes at position d + (S-2)
        # Per-seq: number of branch blocks = blocks needed to cover positions
        # [aligned_down(d), aligned_down(d) + r + max_tail_pos].
        per_seq_branch_blocks: list[int] = []
        for seq in seqs:
            r = seq.d_committed_len % bsz
            span = r + max_tail_pos + 1  # +1 for inclusive count
            n = max(1, (span + bsz - 1) // bsz)
            per_seq_branch_blocks.append(n)
        max_bb = max(per_seq_branch_blocks) if per_seq_branch_blocks else 1

        prefix_aligned_blocks = [
            (s.d_committed_len + bsz - 1) // bsz for s in seqs
        ]

        # block_table per (seq, branch) length:
        #   prefix_blocks_truncated ++ branch_blocks
        # where prefix_blocks_truncated drops the last prefix block IF the
        # branch needs to own the partial slot (i.e. r > 0 OR we need to
        # extend); the dropped slot's content is copied into branch_blocks[0].
        max_prefix_kept = max(
            (s.d_committed_len // bsz) for s in seqs
        )  # full blocks before the partial one
        max_blocks_branch = max_prefix_kept + max_bb

        bt_branch_np = np.full((bs * K, max_blocks_branch), -1, dtype=np.int32)
        branch_blocks_first_np = np.empty((bs, K), dtype=np.int64)

        # Collect (src, dst) block ids for the batched partial-prefix-block
        # copy so we can issue one kernel instead of B*K small kernels.
        copy_src: list[int] = []
        copy_dst: list[int] = []

        for j, seq in enumerate(seqs):
            r = seq.d_committed_len % bsz
            n_full_prefix = seq.d_committed_len // bsz
            n_branch = per_seq_branch_blocks[j]
            full_prefix = seq.d_blocks[:n_full_prefix]

            # Allocate K * n_branch fresh blocks for this seq.
            new_blocks = self.draft_kv.alloc(K * n_branch)
            seq._chain_tail_blocks = new_blocks

            need_partial_copy = r > 0 and len(seq.d_blocks) > n_full_prefix
            partial_src = int(seq.d_blocks[n_full_prefix]) if need_partial_copy else -1

            for k in range(K):
                bb = new_blocks[k * n_branch:(k + 1) * n_branch]
                if need_partial_copy:
                    copy_src.append(partial_src)
                    copy_dst.append(bb[0])
                base = j * K + k
                bt_branch_np[base, :n_full_prefix] = full_prefix
                bt_branch_np[base, n_full_prefix:n_full_prefix + n_branch] = bb
                branch_blocks_first_np[j, k] = bb[0]

        if copy_src:
            src_t = torch.tensor(copy_src, device=self.device, dtype=torch.long)
            dst_t = torch.tensor(copy_dst, device=self.device, dtype=torch.long)
            self.draft_kv.kv[:, :, dst_t] = self.draft_kv.kv[:, :, src_t]

        bt_branch = torch.from_numpy(bt_branch_np).to(self.device)
        branch_blocks_first = torch.from_numpy(branch_blocks_first_np).to(
            self.device,
        )                                                                  # [B, K]

        base_pos_np = np.array(
            [s.d_committed_len for s in seqs], dtype=np.int64,
        )
        base_pos_bk = (
            torch.from_numpy(base_pos_np).to(self.device).repeat_interleave(K)
        )                                                                  # [B*K]
        d_committed_t = torch.from_numpy(base_pos_np).to(self.device)      # [B]
        # offset within the FIRST branch block where the tail begins.
        r_per_seq = torch.from_numpy(base_pos_np % bsz).to(self.device)    # [B]
        # Pre-computed CPU-side max base_pos so we can derive max_ctx for each
        # chain step without a GPU sync per step.
        max_base_pos_cpu = int(base_pos_np.max())

        for f in range(S - 1):
            # Position for this forward step: base_pos + f for every branch.
            positions = base_pos_bk + f                                    # [B*K]

            # Slot mapping: branch's first block * block_size + (r + f), for
            # the case where the tail still fits in the first branch block.
            # If (r + f) >= block_size we'd need to look at branch_blocks[1+...];
            # we restrict S so that r + (S-2) < 2*block_size at most and
            # allocate enough branch blocks above; here pick the right block.
            offset_full = r_per_seq + f                                    # [B]
            block_idx_within_branch = (offset_full // bsz).to(torch.long)  # [B]
            offset_in_block = (offset_full % bsz).to(torch.long)           # [B]
            # Resolve the block id for each (seq, branch) at this f.
            # The branch's blocks live at bt_branch[base, n_full_prefix..]
            # so the i-th branch block id can be computed from
            # branch_blocks_first when block_idx == 0 (the common case) or
            # by gathering from bt_branch otherwise.
            # For simplicity, always gather from bt_branch.
            n_full_prefix_t = (d_committed_t // bsz).to(torch.long)        # [B]
            block_table_col = (
                n_full_prefix_t + block_idx_within_branch
            ).repeat_interleave(K)                                         # [B*K]
            block_ids_at_f = bt_branch[
                torch.arange(bs * K, device=self.device), block_table_col,
            ].to(torch.int64)                                              # [B*K]
            slot_mapping = (
                block_ids_at_f * bsz
                + offset_in_block.repeat_interleave(K).to(torch.int64)
            ).to(torch.int32)                                              # [B*K]

            # cache_seqlens for decode-mode attention = base_pos + f + 1.
            cache_seqlens = (d_committed_t + (f + 1)).to(torch.int32)      # [B]
            cache_seqlens_bk = cache_seqlens.repeat_interleave(K)          # [B*K]
            max_ctx = max_base_pos_cpu + (f + 1)

            with set_forward_context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=cache_seqlens_bk,
                block_tables=bt_branch,
                max_context_len=max_ctx,
            ):
                hidden_to_logits, hidden_to_aux = self.draft.forward_draft(
                    input_ids_bk, positions.to(torch.long), hidden_bk,
                )
                # Decode-mode: ParallelLMHead.project returns one logit per
                # query, so this gives [B*K, draft_vocab].
                logits = self.draft.compute_logits(hidden_to_logits)

            log_probs = torch.log_softmax(logits, dim=-1)                  # [B*K, V_d]
            topk_p, topk_index = torch.topk(log_probs, K, dim=-1)          # [B*K, K]
            topk_index_t = self.draft.remap_draft_ids(topk_index)          # [B*K, K]

            # select_top_k: combine cumulative log-prob scores (B, K) with new
            # K*K candidate log-probs and pick top K per seq. Sglang multiplies
            # probabilities; we add log-probs which is mathematically equivalent
            # for ordering and avoids underflow on long chains.
            expand_scores = (
                scores.unsqueeze(2) + topk_p.reshape(bs, K, K)
            )                                                              # [B, K, K]
            topk_cs_p, topk_cs_index = torch.topk(
                expand_scores.flatten(start_dim=1), K, dim=-1,
            )                                                              # [B, K]
            scores = topk_cs_p

            topk_index_t_flat = topk_index_t.reshape(bs, K * K)            # [B, K*K]
            new_input_ids = torch.gather(
                topk_index_t_flat, dim=1, index=topk_cs_index,
            ).reshape(bs * K)                                              # [B*K]

            # Gather selected hidden states for the next step (per branch).
            sel_idx = (
                topk_cs_index.flatten() // K
                + torch.arange(0, bs * K, K, device=self.device)
                .repeat_interleave(K)
            )                                                              # [B*K]
            new_hidden = hidden_to_aux[sel_idx]

            # Tree-info bookkeeping for organize_draft_results.
            score_list.append(expand_scores)                               # [B, K, K]
            token_list.append(topk_index_t_flat)                           # [B, K*K]
            parents_list.append(topk_cs_index + (K * K * f + K))            # [B, K]

            input_ids_bk = new_input_ids
            hidden_bk = new_hidden

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.num_draft_tokens,
        )
        return parent_list, top_scores_index, draft_tokens, verified_id

    # ------------------------------------------------------------------
    # Verify forward on the target
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _build_tree_cascade_metadata(
        self,
        prefix_lens_cpu: List[int],
        slot_mapping_draft: torch.Tensor,
        tree_mask: torch.Tensor,
    ) -> dict:
        """Build the FA3 cascade-attention metadata for ``_target_verify``.

        Mirrors sglang's ``target_verify_metadata_topk_normal`` (prefix pass)
        and ``target_verify_metadata_topk_expand`` (per-query draft pass) in
        ``flashattention_backend.py``.

        Inputs
        ------
        prefix_lens_cpu : list[int]  -- length B; ``t_committed_len`` per seq.
        slot_mapping_draft : [B*N] int32  -- token-level cache slot for each
            draft token in linear write order, slot[i*N + k] = slot for seq i,
            draft token k.
        tree_mask : flat bool [seq_lens_sum*N + N*N*B] in FULL_MASK layout.

        Returns
        -------
        dict with the seven tensors / scalars consumed by ``TreeAttnPrefill``.
        """
        device = self.device
        N = self.num_draft_tokens
        bs = len(prefix_lens_cpu)

        prefix_lens = torch.tensor(prefix_lens_cpu, device=device, dtype=torch.long)

        cu_seqlens_q_prefix = torch.arange(
            0, (bs + 1) * N, N, device=device, dtype=torch.int32,
        )
        cache_seqlens_prefix = prefix_lens.to(torch.int32)
        max_seqlen_k_prefix = max(prefix_lens_cpu) if prefix_lens_cpu else 0

        # Per-query draft slot list: [bs*N, N], one row per query, each row
        # contains the N draft slot indices of the query's sequence.
        draft_slots_per_seq = slot_mapping_draft.view(bs, N).long()
        draft_slots_per_query = draft_slots_per_seq.repeat_interleave(N, dim=0)

        # Extract the N draft-side bits from the FULL_MASK tree_mask for each
        # query. For seq i, query j the bits live at:
        #   start_ij = sum_{i'<i} N*(prefix_lens[i'] + N)
        #              + j * (prefix_lens[i] + N) + prefix_lens[i]
        # spanning N consecutive bool entries.
        row_lens = prefix_lens + N
        seq_total = N * row_lens
        seq_offsets = torch.zeros(bs, device=device, dtype=torch.long)
        if bs > 1:
            seq_offsets[1:] = torch.cumsum(seq_total[:-1], dim=0)

        i_idx = torch.arange(bs, device=device).repeat_interleave(N)  # [bs*N]
        j_idx = torch.arange(N, device=device).repeat(bs)             # [bs*N]

        start_offsets = (
            seq_offsets[i_idx]
            + j_idx * row_lens[i_idx]
            + prefix_lens[i_idx]
        )                                                              # [bs*N]
        col_offsets = torch.arange(N, device=device, dtype=torch.long) # [N]
        mask_indices = start_offsets[:, None] + col_offsets[None, :]   # [bs*N, N]
        draft_mask = tree_mask[mask_indices].to(torch.bool)            # [bs*N, N]

        # Sort each row so the True entries come first (match sglang).
        col_keys = col_offsets.expand(bs * N, -1)
        keys = torch.where(draft_mask, col_keys, col_keys + N)         # [bs*N, N]
        _, sort_order = torch.sort(keys, dim=1)

        page_table_expand = (
            draft_slots_per_query.gather(1, sort_order).to(torch.int32)
        )                                                              # [bs*N, N]
        cache_seqlens_expand = draft_mask.sum(dim=1).to(torch.int32)   # [bs*N]
        cu_seqlens_q_expand = torch.arange(
            bs * N + 1, device=device, dtype=torch.int32,
        )

        return {
            "cu_seqlens_q_prefix": cu_seqlens_q_prefix,
            "cache_seqlens_prefix": cache_seqlens_prefix,
            "max_seqlen_k_prefix": max_seqlen_k_prefix,
            "page_table_expand": page_table_expand,
            "cache_seqlens_expand": cache_seqlens_expand,
            "cu_seqlens_q_expand": cu_seqlens_q_expand,
        }

    @torch.inference_mode()
    def _target_verify(
        self,
        seqs: List[_Eagle3Sequence],
        full_draft_tokens: torch.Tensor,
        tree_mask: torch.Tensor,
        positions_tensor: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_next_token: torch.Tensor,
        retrive_next_sibling: torch.Tensor,
    ):
        bs = len(seqs)
        N = self.num_draft_tokens

        slot_mapping: list[int] = []
        block_tables_rows: list[list[int]] = []
        prefix_lens_cpu: list[int] = []
        for seq in seqs:
            new_total = seq.t_committed_len + N
            self._ensure_target_capacity(seq, new_total)
            slot_mapping.extend(
                self._slot_mapping_for_range(
                    seq.t_blocks, seq.t_committed_len, new_total,
                )
            )
            block_tables_rows.append(seq.t_blocks)
            prefix_lens_cpu.append(seq.t_committed_len)

        max_blocks = max(len(b) for b in block_tables_rows)
        bt = np.full((bs, max_blocks), -1, dtype=np.int32)
        for i, b in enumerate(block_tables_rows):
            bt[i, :len(b)] = b

        slot_t = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        bt_t = torch.from_numpy(bt).to(self.device)
        # Total context length per seq (prefix + N draft tokens). This is the
        # legacy ``context_lens`` field; the FA3 cascade uses
        # ``cache_seqlens_prefix`` (prefix only) and ``cache_seqlens_expand``
        # (per-query attended draft count) instead.
        ctx_lens_cpu = [p + N for p in prefix_lens_cpu]
        ctx_t = torch.tensor(ctx_lens_cpu, dtype=torch.int32, device=self.device)
        max_ctx_cpu = max(ctx_lens_cpu)

        cascade = self._build_tree_cascade_metadata(
            prefix_lens_cpu, slot_t, tree_mask,
        )

        with set_forward_context(
            is_prefill=False,
            slot_mapping=slot_t,
            context_lens=ctx_t,
            block_tables=bt_t,
            max_context_len=max_ctx_cpu,
            is_tree_verify=True,
            tree_num_verify_tokens=N,
            tree_block_table_prefix=bt_t,
            tree_cache_seqlens_prefix=cascade["cache_seqlens_prefix"],
            tree_cu_seqlens_q_prefix=cascade["cu_seqlens_q_prefix"],
            tree_max_seqlen_q_prefix=N,
            tree_max_seqlen_k_prefix=cascade["max_seqlen_k_prefix"],
            tree_page_table_expand=cascade["page_table_expand"],
            tree_cache_seqlens_expand=cascade["cache_seqlens_expand"],
            tree_cu_seqlens_q_expand=cascade["cu_seqlens_q_expand"],
        ):
            ids_t = full_draft_tokens.to(torch.int64)
            out = self.target.model(ids_t, positions_tensor.to(torch.long))
            if isinstance(out, tuple):
                hidden_states, aux_list = out
            else:
                hidden_states, aux_list = out, []

            # compute_logits while still in is_prefill=False context so it
            # returns one logit per *flat tree token* (shape [B*N, vocab]).
            logits = self.target.compute_logits(hidden_states)
        target_predict = logits.argmax(dim=-1)  # [B*N]

        candidates = full_draft_tokens.view(bs, N).to(torch.long)
        target_predict_2d = target_predict.view(bs, N).to(torch.long)
        predicts = torch.full((bs * N,), -1, device=self.device, dtype=torch.int32)
        accept_index = torch.full(
            (bs, self.spec_steps + 2), -1, device=self.device, dtype=torch.int32,
        )
        accept_token_num = torch.zeros(bs, device=self.device, dtype=torch.int32)

        verify_tree_greedy(
            predicts, accept_index, accept_token_num,
            candidates,
            retrive_index.view(bs, N).to(torch.long),
            retrive_next_token.view(bs, N).to(torch.long),
            retrive_next_sibling.view(bs, N).to(torch.long),
            target_predict_2d,
        )

        return predicts, accept_index, accept_token_num, hidden_states, aux_list

    # ------------------------------------------------------------------
    # Single EAGLE-3 step: draft chain -> tree build -> verify -> accept
    # ------------------------------------------------------------------
    def _eagle3_step(self, seqs: List[_Eagle3Sequence], eos_id: int):
        import os as _os
        _dbg = _os.environ.get("KB_NANO_EAGLE3_DEBUG") == "1"
        if _dbg:
            print(f"[dbg] step start, n_active={len(seqs)} "
                  f"committed_lens={[s.t_committed_len for s in seqs]} "
                  f"d_committed_lens={[s.d_committed_len for s in seqs]}", flush=True)
        # 1. Draft chain (uses each seq's draft_topk_* + draft_hidden cached
        #    by the most recent draft-extend).
        parent_list, top_scores_index, draft_tokens, verified_id = self._draft_chain(seqs)
        if _dbg:
            print(f"[dbg] draft chain done", flush=True)

        # 2. Build tree.
        seq_lens_cpu_list = [s.t_committed_len for s in seqs]
        seq_lens = torch.tensor(
            seq_lens_cpu_list, device=self.device, dtype=torch.long,
        )
        seq_lens_sum = sum(seq_lens_cpu_list)
        (
            tree_mask, positions, retrive_index, retrive_next_token,
            retrive_next_sibling, full_draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            topk=self.topk,
            spec_steps=self.spec_steps,
            num_verify_tokens=self.num_draft_tokens,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
        )

        if _dbg:
            print(f"[dbg] tree built, full_draft_tokens={full_draft_tokens.shape}", flush=True)

        # 3. Verify.
        (
            predicts, accept_index, accept_token_num, hidden_states, aux_list,
        ) = self._target_verify(
            seqs, full_draft_tokens, tree_mask, positions,
            retrive_index, retrive_next_token, retrive_next_sibling,
        )
        if _dbg:
            print(f"[dbg] verify done, accept_token_num={accept_token_num.tolist()}, "
                  f"accept_index[0]={accept_index[0].tolist()}, "
                  f"committed_pre={[s.t_committed_len for s in seqs]}", flush=True)

        # 4. Accept tokens; roll back rejected target KV.
        N = self.num_draft_tokens
        accept_num_cpu = accept_token_num.tolist()
        accept_index_cpu = accept_index.tolist()

        # Remap target K/V so that logical position ``t_committed_len + k``
        # holds the K/V for the k-th accepted token (NOT for the k-th tree
        # node in flat write order). Must run before we touch t_committed_len
        # or free any blocks. With chain drafting this is a no-op; with tree
        # drafting it is required for correctness.
        self._remap_target_kv_after_verify(seqs, accept_index_cpu, accept_num_cpu)

        if len(aux_list) > 0:
            aux_cat_full = torch.cat(aux_list, dim=-1)  # [B*N, 3*H_t]
        else:
            aux_cat_full = None

        # Per-seq accepted tokens + aux for the post-verify draft extend.
        accepted_lists: list[list[int]] = []
        accepted_aux_per_seq: list[torch.Tensor] = []

        for i, seq in enumerate(seqs):
            n_accept = int(accept_num_cpu[i])  # # accepted SPECULATIVE
            accepted_indices_flat = accept_index_cpu[i][: n_accept + 1]
            accepted_ids = [int(predicts[idx]) for idx in accepted_indices_flat]
            seq.token_ids.extend(accepted_ids)
            seq.generated_ids.extend(accepted_ids)
            seq.last_token = accepted_ids[-1]
            accepted_lists.append(accepted_ids)

            tentative_committed = seq.t_committed_len + N
            new_committed = seq.t_committed_len + (n_accept + 1)
            old_blocks = (tentative_committed + self.block_size - 1) // self.block_size
            new_blocks = (new_committed + self.block_size - 1) // self.block_size
            if new_blocks < old_blocks:
                released = seq.t_blocks[new_blocks:]
                self.target_kv.free(released)
                seq.t_blocks = seq.t_blocks[:new_blocks]
            seq.t_committed_len = new_committed

            if aux_cat_full is not None:
                acc_aux = aux_cat_full[
                    torch.tensor(accepted_indices_flat, device=self.device,
                                 dtype=torch.long)
                ]                                          # [n_accept+1, 3*H_t]
                accepted_aux_per_seq.append(acc_aux)
                seq.last_aux = acc_aux[-1].clone()
            else:
                accepted_aux_per_seq.append(torch.zeros(
                    (len(accepted_ids), self.target_config.hidden_size * 3),
                    device=self.device, dtype=self.dtype,
                ))

            # Free the per-branch tail blocks allocated by _draft_chain.
            # The draft KV will be re-extended with the actually-accepted
            # tokens (and target aux hidden states) below.
            tail = getattr(seq, "_chain_tail_blocks", None)
            if tail is not None and len(tail) > 0:
                self.draft_kv.free(list(tail))
                seq._chain_tail_blocks = []
            # Also drop any stray prefix-tail blocks (none should exist after
            # the prior extend, but keep this for safety).
            self._free_draft_tail(seq, seq.d_committed_len)

            if not seq.ignore_eos and (eos_id in accepted_ids):
                pos = accepted_ids.index(eos_id)
                drop = len(accepted_ids) - (pos + 1)
                if drop > 0:
                    del seq.token_ids[-drop:]
                    del seq.generated_ids[-drop:]
                seq.finished = True
            elif len(seq.generated_ids) >= seq.max_tokens:
                if len(seq.generated_ids) > seq.max_tokens:
                    excess = len(seq.generated_ids) - seq.max_tokens
                    del seq.token_ids[-excess:]
                    del seq.generated_ids[-excess:]
                seq.finished = True

        # 5. Post-verify draft extend: digest the accepted tokens into draft KV
        #    and capture (topk_p, topk_i, hidden) for the next chain step 0.
        live_seqs: list[_Eagle3Sequence] = []
        ext_input_ids: list[torch.Tensor] = []
        ext_hiddens: list[torch.Tensor] = []
        ext_lens: list[int] = []
        for i, seq in enumerate(seqs):
            if seq.finished:
                continue
            acc_ids = accepted_lists[i]
            acc_aux = accepted_aux_per_seq[i]
            # sglang ``prepare_extend_after_decode``: extend the draft over the
            # M+1 accepted positions. Input ids are SHIFTED LEFT BY 1, i.e.
            # for accepted tokens [a_0, a_1, ..., a_M], input = [a_1, ..., a_M, last].
            # The "last" input is the last accepted token (per
            # create_extend_after_decode_spec_info, verified_id_data is
            # accepted_ids[-1] for the next round, but for THIS extend the
            # input at the final position needs to be the new bonus = a_M).
            # In sglang's after-decode extend, batch.input_ids = self.verified_id
            # which contains exactly accept_length per seq slots, with the
            # LAST slot being the new verified id (= last accepted). So the
            # input_ids at the M+1 extend positions are [a_0, a_1, ..., a_M].
            # But that's NOT shifted - matches their `prepare_extend_after_decode`
            # which sets `batch.input_ids = self.verified_id` directly.
            #
            # And `verified_id` is filled per-seq with `verified_id_data` =
            # the value at `accept_len_cumsum + accept_length - 1` of the
            # original verified_id buffer. Hmm - that's just the last accepted.
            # For the EXTEND it actually uses `self.verified_id` after the kernel
            # which fills it with one entry per seq = last accepted; so the
            # input at the last extend position is the last accepted. The
            # earlier positions are filled by the triton kernel writing into
            # `batch.input_ids` based on the same buffer... let me trust
            # sglang semantics: input at draft position d_committed_len + j
            # (j = 0..M) should be accepted_ids[j], and the draft model
            # then predicts what comes after the LAST accepted token.
            ids = torch.tensor(acc_ids, dtype=torch.int64, device=self.device)
            ext_input_ids.append(ids)
            ext_hiddens.append(acc_aux)
            ext_lens.append(len(acc_ids))
            live_seqs.append(seq)

        if live_seqs:
            self._draft_extend(live_seqs, ext_input_ids, ext_hiddens, ext_lens)
            if _dbg:
                print(f"[dbg] post-verify draft extend done; "
                      f"d_committed_lens={[s.d_committed_len for s in live_seqs]}",
                      flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        self.target_kv.reset()
        self.draft_kv.reset()

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[List[int]] | List[str],
        sampling_params: Eagle3SamplingParams | List[Eagle3SamplingParams],
        use_tqdm: bool = False,
    ) -> List[Eagle3Output]:
        if isinstance(sampling_params, Eagle3SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        assert len(sampling_params) == len(prompts)

        prompt_token_ids: List[List[int]] = []
        for p in prompts:
            if isinstance(p, str):
                prompt_token_ids.append(
                    self.tokenizer.encode(p, add_special_tokens=False)
                )
            else:
                prompt_token_ids.append(list(p))

        eos_id = self.tokenizer.eos_token_id

        all_outputs: List[Optional[Eagle3Output]] = [None] * len(prompts)
        idx_pool = list(range(len(prompts)))
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=len(prompts), desc="EAGLE-3 generate")
        else:
            pbar = None

        while idx_pool:
            batch_idx = idx_pool[: self.max_num_seqs]
            idx_pool = idx_pool[self.max_num_seqs:]

            seqs = [
                _Eagle3Sequence(
                    prompt_token_ids[i],
                    sampling_params[i].max_tokens,
                    sampling_params[i].ignore_eos,
                )
                for i in batch_idx
            ]
            aux_per_seq = self._target_prefill(seqs)

            # Initial draft EXTEND on the prompt (sglang's
            # ``forward_draft_extend`` for prefill). Input is the prompt
            # shifted left by 1, with the bonus token replacing the final
            # slot:  [tok_1, tok_2, ..., tok_{P-1}, bonus_P].
            ext_input_ids: list[torch.Tensor] = []
            ext_hiddens: list[torch.Tensor] = []
            ext_lens: list[int] = []
            live_for_extend: list[_Eagle3Sequence] = []
            for i, seq in enumerate(seqs):
                P = len(seq.prompt_ids)
                ids = list(seq.prompt_ids[1:]) + [seq.last_token]
                ext_input_ids.append(
                    torch.tensor(ids, dtype=torch.int64, device=self.device)
                )
                ext_hiddens.append(aux_per_seq[i])
                ext_lens.append(P)
                live_for_extend.append(seq)
            if live_for_extend:
                self._draft_extend(
                    live_for_extend, ext_input_ids, ext_hiddens, ext_lens,
                )

            for seq in seqs:
                if not seq.ignore_eos and seq.last_token == eos_id:
                    seq.finished = True
                if len(seq.generated_ids) >= seq.max_tokens:
                    seq.finished = True

            while any(not s.finished for s in seqs):
                active = [s for s in seqs if not s.finished]
                self._eagle3_step(active, eos_id)

            for seq, src_i in zip(seqs, batch_idx):
                if seq.t_blocks:
                    self.target_kv.free(seq.t_blocks)
                    seq.t_blocks.clear()
                if seq.d_blocks:
                    self.draft_kv.free(seq.d_blocks)
                    seq.d_blocks.clear()
                all_outputs[src_i] = Eagle3Output(
                    prompt_token_ids=seq.prompt_ids,
                    token_ids=list(seq.generated_ids),
                    generated_text=self.tokenizer.decode(seq.generated_ids),
                )
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        return all_outputs
