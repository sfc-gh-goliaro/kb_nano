"""EAGLE-3 speculative-decoding engine for kb_nano (eager-only, chain draft).

This first cut implements EAGLE-3 with a *chain* draft (topk=1) instead of the
full topk=8 tree. Greedy chain drafting + greedy verify still produces the
exact same accepted-token sequence as the target model would generate on its
own (because we keep tokens only when they match ``target.argmax``), so it
satisfies the benchmark's per-request token alignment metric. The
``num_draft_tokens = spec_steps`` chain accepts roughly the prefix of matching
tokens at each step, giving real speedups over plain target decode.

Tree drafting (topk=8) for higher mean acceptance can be plugged in later by
replacing ``_draft_chain`` with a per-step tree expansion.

Scope (matches the user's "tree_eager" choice, simplified):
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
        spec_steps: int = 5,
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda")
        self.dtype = dtype
        # Chain drafting: topk = 1, num_draft_tokens = spec_steps + 1
        # (sglang's tree convention: 1 root + S draft = S+1 verified positions).
        self.spec_steps = spec_steps
        self.topk = 1
        self.num_draft_tokens = spec_steps + 1
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
    # Draft chain (topk=K, S steps).  Step 0 only records (no forward);
    # steps 1..S-1 each do one draft forward.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _draft_chain(self, seqs: List[_Eagle3Sequence]):
        """Produce sglang-format tree inputs from S sequential draft steps.

        Step 0 uses the (topk_p, topk_i, hidden) cached on each seq from the
        most recent draft-extend.  Subsequent steps run a 1-query forward and
        write to draft KV at positions ``[d_committed_len .. d_committed_len + S - 2]``.
        """
        S = self.spec_steps
        K = self.topk  # = 1 for chain
        bs = len(seqs)

        verified_id = torch.tensor(
            [s.last_token for s in seqs], device=self.device, dtype=torch.long,
        )

        score_list: list[torch.Tensor] = []
        token_list: list[torch.Tensor] = []
        parents_list: list[torch.Tensor] = []

        # Pull initial state from most recent draft extend.
        cur_top_p = torch.stack([s.draft_topk_p for s in seqs])           # [bs, K]
        cur_top_i = torch.stack([s.draft_topk_i for s in seqs])           # [bs, K]
        cur_top_i_t = torch.stack([s.draft_topk_i_t for s in seqs])       # [bs, K]
        cur_hidden = torch.stack([s.draft_hidden for s in seqs])          # [bs, H_d]

        # Step 0: just record (no forward).
        score_list.append(cur_top_p.unsqueeze(1))                          # [bs, 1, K]
        token_list.append(cur_top_i_t.view(bs, K))                         # [bs, K]
        parents_list.append(
            torch.arange(-1, K, dtype=torch.long, device=self.device)
            .unsqueeze(0).repeat(bs, 1)                                    # [bs, K + 1]
        )

        for step in range(1, S):
            # Shift convention: at draft pos p, input is the token at seq pos
            # p+1 and output predicts seq pos p+2.  cur_top_i_t holds the
            # prediction for seq pos d_committed_len + step + 1 from the
            # previous step's output, so we forward at draft pos
            # d_committed_len + (step - 1).
            slot_mapping: list[int] = []
            positions_list: list[int] = []
            block_tables_rows: list[list[int]] = []
            for seq in seqs:
                pos = seq.d_committed_len + (step - 1)
                self._ensure_draft_capacity(seq, pos + 1)
                slot = seq.d_blocks[pos // self.block_size] * self.block_size + (
                    pos % self.block_size
                )
                slot_mapping.append(slot)
                positions_list.append(pos)
                block_tables_rows.append(seq.d_blocks)

            max_blocks = max(len(b) for b in block_tables_rows)
            bt = np.full((bs, max_blocks), -1, dtype=np.int32)
            for i, b in enumerate(block_tables_rows):
                bt[i, :len(b)] = b

            cu_q = torch.arange(bs + 1, device=self.device, dtype=torch.int32)
            cu_k_vals = [0]
            for seq in seqs:
                cu_k_vals.append(cu_k_vals[-1] + seq.d_committed_len + step)
            cu_k = torch.tensor(cu_k_vals, dtype=torch.int32, device=self.device)
            max_sk = max(seq.d_committed_len + step for seq in seqs)

            set_context(
                True,
                cu_q, cu_k,
                1, max_sk,
                torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
                block_tables=torch.from_numpy(bt).to(self.device),
            )

            ids_t = cur_top_i_t.view(bs).to(torch.int64)
            pos_t = torch.tensor(positions_list, dtype=torch.long, device=self.device)

            hidden_to_logits, hidden_to_aux = self.draft.forward_draft(
                ids_t, pos_t, cur_hidden,
            )
            logits = self.draft.compute_logits(hidden_to_logits)
            log_probs = torch.log_softmax(logits, dim=-1)
            top_p, top_i = torch.topk(log_probs, K, dim=-1)               # [bs, K]
            top_i_t = self.draft.remap_draft_ids(top_i)                    # [bs, K]

            # sglang tree shape for K=1: each subsequent step contributes
            # K=1 new candidate per seq, parented to the previous level's
            # single node.  cs_index is 0 (only 1 sibling).
            score_list.append(top_p.view(bs, K, K))                        # [bs, K, K]
            token_list.append(top_i_t.view(bs, K * K))                     # [bs, K*K]
            cs_index = torch.zeros((bs, K), dtype=torch.long, device=self.device)
            parents_list.append(cs_index + (K * K * (step - 1) + K))       # [bs, K]

            cur_top_p = top_p
            cur_top_i = top_i
            cur_top_i_t = top_i_t
            cur_hidden = hidden_to_aux

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.num_draft_tokens,
        )
        return parent_list, top_scores_index, draft_tokens, verified_id

    # ------------------------------------------------------------------
    # Verify forward on the target
    # ------------------------------------------------------------------
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
        ctx_lens: list[int] = []
        for seq in seqs:
            new_total = seq.t_committed_len + N
            self._ensure_target_capacity(seq, new_total)
            slot_mapping.extend(
                self._slot_mapping_for_range(
                    seq.t_blocks, seq.t_committed_len, new_total,
                )
            )
            block_tables_rows.append(seq.t_blocks)
            ctx_lens.append(new_total)

        max_blocks = max(len(b) for b in block_tables_rows)
        bt = np.full((bs, max_blocks), -1, dtype=np.int32)
        for i, b in enumerate(block_tables_rows):
            bt[i, :len(b)] = b

        slot_t = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        bt_t = torch.from_numpy(bt).to(self.device)
        ctx_t = torch.tensor(ctx_lens, dtype=torch.int32, device=self.device)

        with set_forward_context(
            is_prefill=False,
            slot_mapping=slot_t,
            context_lens=ctx_t,
            block_tables=bt_t,
            max_context_len=int(max(ctx_lens)),
            is_tree_verify=True,
            tree_mask=tree_mask,
            tree_num_verify_tokens=N,
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
        seq_lens = torch.tensor(
            [s.t_committed_len for s in seqs], device=self.device, dtype=torch.long,
        )
        seq_lens_sum = int(seq_lens.sum().item())
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

            # Free the draft KV chain slots we appended during _draft_chain
            # (positions [d_committed_len .. d_committed_len + S - 2]).
            # We then re-extend with the actually-accepted tokens below.
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
