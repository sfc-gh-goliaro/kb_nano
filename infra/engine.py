"""
Batched inference engine with paged KV cache and tensor parallelism.

Architecture closely follows nano-vllm:
  - ModelRunner handles model init, KV cache, CUDA graphs on each GPU
  - For TP>1, rank 0 serializes method calls via shared memory
  - Non-rank-0 workers block in a loop inside ModelRunner.__init__
  - LlamaEngine (rank 0 only) drives scheduling and sampling

No vLLM imports.
"""

from __future__ import annotations

import atexit
import os
import pickle
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from .context import get_context, reset_context, set_context, set_mixed_context
from .gla_state import GLAStateManager
from .kimi_linear_state import KimiLinearStateManager
from .mamba_state import MambaStateManager
from .rwkv7_state import RWKV7StateManager
from ..tasks.baseline.L1.allreduce import set_custom_ar
from .weight_loader import load_model

BLOCK_SIZE = 256
MAX_NUM_BATCHED_TOKENS = 16384
MAX_NUM_SEQS = 512
MAX_MODEL_LEN = 8192
NCCL_PORT = int(os.environ.get("KB_NANO_NCCL_PORT", "29501"))

# Placeholder token IDs for Qwen VL models
QWEN_IMAGE_PAD_ID = 151655  # <|image_pad|>
QWEN_VIDEO_PAD_ID = 151656  # <|video_pad|>

_PROFILE = os.environ.get("KB_NANO_PROFILE", "0") == "1"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = None
    ignore_eos: bool = False


@dataclass
class GenerationOutput:
    prompt: str
    generated_text: str
    token_ids: list[int] = field(default_factory=list)
    logits_history: list[torch.Tensor] | None = None


class SeqStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    RUNNING = auto()
    FINISHED = auto()


# ---------------------------------------------------------------------------
# Sequence — must be picklable for shared memory transfer
# ---------------------------------------------------------------------------
class Sequence:
    _next_id = 0

    def __init__(self, prompt_ids: list[int], max_tokens: int = 512,
                 ignore_eos: bool = False):
        self.seq_id = Sequence._next_id
        Sequence._next_id += 1
        self.prompt_ids = list(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.generated_ids: list[int] = []
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.block_table: list[int] = []
        self.status = SeqStatus.WAITING
        self.num_computed_tokens: int = 0
        self.state_slot: int | None = None
        # Multimodal fields
        self.pixel_values = None  # preprocessed image pixels
        self.image_grid_thw = None  # list of [t, h, w] per image
        self.video_pixel_values = None
        self.video_grid_thw = None
        self.mrope_position_delta: int = 0
        self.mrope_positions = None  # (3, seq_len) tensor computed at prefill

    def __len__(self):
        if self.token_ids is not None:
            return len(self.token_ids)
        return self._num_tokens

    @property
    def num_blocks(self):
        return (len(self) + BLOCK_SIZE - 1) // BLOCK_SIZE

    @property
    def last_block_num_tokens(self):
        r = len(self) % BLOCK_SIZE
        return r if r else BLOCK_SIZE

    @property
    def last_token(self):
        if self.token_ids is not None:
            return self.token_ids[-1]
        return self._last_token

    @property
    def num_prompt_tokens(self):
        return len(self.prompt_ids)

    @property
    def num_remaining_prefill(self):
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    def blocks_needed_for(self, num_tokens):
        """Number of NEW blocks needed to store num_tokens more KV slots."""
        total_after = self.num_computed_tokens + num_tokens
        blocks_after = (total_after + BLOCK_SIZE - 1) // BLOCK_SIZE
        return max(0, blocks_after - len(self.block_table))

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.generated_ids.append(token_id)

    def __getstate__(self):
        """Minimal pickling for shared memory transfer to non-rank-0 workers."""
        return (
            len(self),
            len(self.prompt_ids),
            self.block_table,
            self.num_computed_tokens,
            self.state_slot,
            self.token_ids if not self.generated_ids else self.last_token,
        )

    def __setstate__(self, state):
        if len(state) == 6:
            self._num_tokens, num_prompt, self.block_table, self.num_computed_tokens, self.state_slot = state[:-1]
        else:
            # Backward-compatible fallback for older pickled tuples.
            self._num_tokens, num_prompt, self.block_table, self.num_computed_tokens = state[:-1]
            self.state_slot = None
        if isinstance(state[-1], list):
            self.token_ids = state[-1]
        else:
            self.token_ids = None
            self._last_token = state[-1]
        self.prompt_ids = []
        self.generated_ids = []


# ---------------------------------------------------------------------------
# Block Manager
# ---------------------------------------------------------------------------
class BlockManager:
    def __init__(self, num_blocks: int):
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def can_allocate(self, seq):
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq):
        for _ in range(seq.num_blocks):
            seq.block_table.append(self.free_block_ids.popleft())

    def can_allocate_n(self, n_blocks):
        return len(self.free_block_ids) >= n_blocks

    def allocate_n(self, seq, n_blocks):
        for _ in range(n_blocks):
            seq.block_table.append(self.free_block_ids.popleft())

    def can_append(self, seq):
        return len(self.free_block_ids) >= (len(seq) % BLOCK_SIZE == 1)

    def may_append(self, seq):
        if len(seq) % BLOCK_SIZE == 1:
            seq.block_table.append(self.free_block_ids.popleft())

    def deallocate(self, seq):
        self.free_block_ids.extend(seq.block_table)
        seq.block_table.clear()


# ---------------------------------------------------------------------------
# ModelRunner — runs on EACH TP rank
# ---------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, model_name: str, rank: int, world_size: int,
                 dtype: torch.dtype, enforce_eager: bool,
                 event, shm_name: str):
        self.rank = rank
        self.world_size = world_size
        self.enforce_eager = enforce_eager
        self.event = event
        self.block_size = BLOCK_SIZE
        self._model_dtype = dtype

        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", f"tcp://localhost:{NCCL_PORT}",
            world_size=world_size, rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )

        self.custom_ar = None
        if world_size > 1:
            self.cpu_group = dist.new_group(backend="gloo")
            if not os.environ.get("KB_NANO_DISABLE_CUSTOM_AR", "0") == "1":
                from ..tasks.baseline.L1.allreduce import CustomAllreduce
                self.custom_ar = CustomAllreduce(
                    self.cpu_group, rank, max_size=8 * 1024 * 1024
                )
                set_custom_ar(self.custom_ar)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")

        self.model, self.config = load_model(
            model_name, torch.device(f"cuda:{rank}"), dtype,
        )
        model_type = getattr(self.config, "model_type", "")
        self.model_family = (
            "mamba" if model_type in {"mamba", "mamba2", "gla", "rwkv7", "kimi_linear", "qwen3_next"}
            else "attention"
        )
        self.is_moe = hasattr(self.config, "num_local_experts")
        self.is_qwen_vl = hasattr(self.config, "mrope_section")
        if self.model_family == "attention":
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()
            self._init_greedy_buffers()
        else:
            self.allocate_mamba_state_cache()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # TP shared memory setup
        if world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=shm_name, create=True, size=2**20)
                self.shm.buf[self._SHM_FLAG_OFFSET] = 0
                self.shm.buf[self._SHM_SEQ_OFFSET:self._SHM_SEQ_OFFSET+4] = (0).to_bytes(4, "little")
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=shm_name)
                self.loop()  # Non-rank-0 blocks here forever

    def exit(self):
        if self.custom_ar is not None:
            self.custom_ar.close()
            self.custom_ar = None
            set_custom_ar(None)
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # SHM layout for spin-wait signaling:
    # byte[-1] (_SHM_FLAG_OFFSET): 0=generic, 1=decode_greedy, 2=exit marker
    # bytes[-5:-1] (_SHM_SEQ_OFFSET): 4-byte little-endian sequence counter
    _SHM_FLAG_OFFSET = 2**20 - 1
    _SHM_SEQ_OFFSET = 2**20 - 5

    def loop(self):
        """Worker loop: spin-wait on SHM sequence counter for decode, event for generic."""
        buf = self.shm.buf
        flag_off = self._SHM_FLAG_OFFSET
        seq_off = self._SHM_SEQ_OFFSET
        last_seq = int.from_bytes(buf[seq_off:seq_off+4], "little")
        while True:
            cur_seq = int.from_bytes(buf[seq_off:seq_off+4], "little")
            if cur_seq != last_seq:
                last_seq = cur_seq
                if buf[flag_off] != 0:
                    self._loop_decode_greedy()
                    continue
                n = int.from_bytes(buf[0:4], "little")
                method_name, *args = pickle.loads(buf[4:n+4])
                getattr(self, method_name)(*args)
                if method_name == "exit":
                    break
                continue
            # Yield briefly to avoid pure busy-wait burning power
            pass

    def _signal_workers(self):
        """Increment SHM sequence counter to wake spin-waiting workers."""
        buf = self.shm.buf
        seq_off = self._SHM_SEQ_OFFSET
        cur = int.from_bytes(buf[seq_off:seq_off+4], "little")
        nxt = (cur + 1) & 0xFFFFFFFF
        buf[seq_off:seq_off+4] = nxt.to_bytes(4, "little")

    def call(self, method_name, *args):
        """Called by rank 0 to execute method on ALL ranks."""
        if self.world_size > 1 and self.rank == 0:
            data = pickle.dumps([method_name, *args])
            n = len(data)
            buf = self.shm.buf
            buf[0:4] = n.to_bytes(4, "little")
            buf[4:n+4] = data
            buf[self._SHM_FLAG_OFFSET] = 0  # generic path
            self._signal_workers()
        return getattr(self, method_name)(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        num_seqs = min(MAX_NUM_BATCHED_TOKENS // MAX_MODEL_LEN, MAX_NUM_SEQS)
        seqs = [Sequence([0] * MAX_MODEL_LEN) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = self.config.num_key_value_heads // self.world_size
        head_dim = self.config.head_dim
        num_layers = self.config.num_hidden_layers
        elem_size = torch.finfo(torch.get_default_dtype()).bits // 8
        block_bytes = 2 * num_layers * BLOCK_SIZE * num_kv_heads * head_dim * elem_size
        num_blocks = int(total * 0.9 - used - peak + current) // block_bytes
        assert num_blocks > 0, f"Not enough GPU memory for KV cache on rank {self.rank}"
        self.num_blocks = num_blocks
        if self.rank == 0:
            print(f"  KV cache: {num_blocks} blocks x {BLOCK_SIZE} = {num_blocks * BLOCK_SIZE} token slots")

        self.kv_cache = torch.empty(
            2, num_layers, num_blocks, BLOCK_SIZE, num_kv_heads, head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def allocate_mamba_state_cache(self):
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        model_type = getattr(self.config, "model_type", "")
        num_layers = self.config.num_hidden_layers

        if model_type == "kimi_linear":
            # Kimi-Linear: lightweight slot-based state (dicts populated at runtime)
            num_slots = MAX_NUM_SEQS
            self.mamba_state_manager = KimiLinearStateManager(
                num_hidden_layers=num_layers,
                num_slots=num_slots,
            )
            self.num_state_slots = num_slots
            if self.rank == 0:
                print(f"  Kimi-Linear state cache: {num_slots} sequence slots")
            return

        if model_type == "qwen3_next":
            # Qwen3-Next: lightweight slot-based state (dicts populated at runtime)
            # Reuse KimiLinearStateManager for simplicity
            num_slots = MAX_NUM_SEQS
            self.mamba_state_manager = KimiLinearStateManager(
                num_hidden_layers=num_layers,
                num_slots=num_slots,
            )
            self.num_state_slots = num_slots
            if self.rank == 0:
                print(f"  Qwen3-Next state cache: {num_slots} sequence slots")
            return

        if model_type == "gla":
            num_heads = self.config.num_heads
            key_dim = int(self.config.hidden_size * self.config.expand_k)
            value_dim = int(self.config.hidden_size * self.config.expand_v)
            head_k_dim = key_dim // num_heads
            head_v_dim = value_dim // num_heads
            state_dtype = torch.float32
            elem_size = torch.finfo(state_dtype).bits // 8
            per_layer_bytes = num_heads * head_k_dim * head_v_dim * elem_size
            per_slot_bytes = num_layers * per_layer_bytes
            budget = int(total * 0.9 - used - peak + current)
            num_slots = max(1, min(MAX_NUM_SEQS, budget // max(1, per_slot_bytes)))

            self.mamba_state_manager = GLAStateManager(
                num_hidden_layers=num_layers,
                num_heads=num_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                num_slots=num_slots,
                dtype=state_dtype,
                device=torch.device(f"cuda:{self.rank}"),
            )
            self.num_state_slots = num_slots
            if self.rank == 0:
                print(f"  GLA state cache: {num_slots} sequence slots")
            return

        if model_type == "rwkv7":
            num_heads = int(self.config.num_heads)
            head_dim = int(self.config.head_dim)
            value_dims = getattr(self.config, "value_dim", None)
            if value_dims is None:
                value_dims = [self.config.hidden_size] * num_layers
            elif isinstance(value_dims, int):
                value_dims = [value_dims] * num_layers
            head_v_dims = [int(v // num_heads) for v in value_dims]

            conv_elem_size = torch.finfo(self._model_dtype).bits // 8
            recurrent_elem_size = torch.finfo(torch.float32).bits // 8
            conv_bytes = num_layers * 2 * self.config.hidden_size * conv_elem_size
            recurrent_bytes = sum(
                num_heads * head_dim * head_v_dim * recurrent_elem_size
                for head_v_dim in head_v_dims
            )
            per_slot_bytes = conv_bytes + recurrent_bytes
            budget = int(total * 0.9 - used - peak + current)
            num_slots = max(1, min(MAX_NUM_SEQS, budget // max(1, per_slot_bytes)))

            self.mamba_state_manager = RWKV7StateManager(
                num_hidden_layers=num_layers,
                hidden_size=self.config.hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                head_v_dims=head_v_dims,
                num_slots=num_slots,
                conv_dtype=self._model_dtype,
                recurrent_dtype=torch.float32,
                device=torch.device(f"cuda:{self.rank}"),
            )
            self.num_state_slots = num_slots
            if self.rank == 0:
                print(f"  RWKV7 state cache: {num_slots} sequence slots")
            return

        elem_size = torch.finfo(self._model_dtype).bits // 8
        if model_type == "mamba2":
            intermediate_size = getattr(
                self.config,
                "intermediate_size",
                int(self.config.expand * self.config.hidden_size),
            )
            conv_dim = intermediate_size + 2 * self.config.n_groups * self.config.state_size
            ssm_state_shape = (
                self.config.num_heads,
                self.config.head_dim,
                self.config.state_size,
            )
            per_layer_bytes = (
                conv_dim * self.config.conv_kernel
                + self.config.num_heads * self.config.head_dim * self.config.state_size
            ) * elem_size
        else:
            conv_dim = self.config.intermediate_size
            ssm_state_shape = (self.config.intermediate_size, self.config.state_size)
            per_layer_bytes = (
                conv_dim * self.config.conv_kernel
                + self.config.intermediate_size * self.config.state_size
            ) * elem_size

        per_slot_bytes = num_layers * per_layer_bytes
        budget = int(total * 0.9 - used - peak + current)
        num_slots = max(1, min(MAX_NUM_SEQS, budget // max(1, per_slot_bytes)))

        self.mamba_state_manager = MambaStateManager(
            num_hidden_layers=num_layers,
            conv_dim=conv_dim,
            ssm_state_shape=ssm_state_shape,
            conv_kernel=self.config.conv_kernel,
            num_slots=num_slots,
            dtype=self._model_dtype,
            device=torch.device(f"cuda:{self.rank}"),
        )
        self.num_state_slots = num_slots
        if self.rank == 0:
            print(f"  Mamba state cache: {num_slots} sequence slots")

    def can_allocate_mamba_state(self):
        return self.mamba_state_manager.has_free_slot()

    def allocate_mamba_state(self, seq):
        self.mamba_state_manager.allocate(seq)

    def deallocate_mamba_state(self, seq):
        self.mamba_state_manager.deallocate(seq)

    def prepare_prefill(self, seqs):
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_sq, max_sk = 0, 0
        slot_mapping = []
        max_bt = 0
        use_mrope = self.is_qwen_vl and any(s.mrope_positions is not None for s in seqs)
        mrope_pos_list = [] if use_mrope else None

        for seq in seqs:
            sl = len(seq)
            input_ids.extend(seq.token_ids)
            if use_mrope and seq.mrope_positions is not None:
                mrope_pos_list.append(seq.mrope_positions)
            else:
                positions.extend(range(sl))
                if use_mrope:
                    mrope_pos_list.append(
                        torch.arange(sl, dtype=torch.int64).unsqueeze(0).expand(3, -1)
                    )
            cu_seqlens_q.append(cu_seqlens_q[-1] + sl)
            cu_seqlens_k.append(cu_seqlens_k[-1] + sl)
            max_sq = max(sl, max_sq)
            max_sk = max(sl, max_sk)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_blocks):
                start = seq.block_table[i] * BLOCK_SIZE
                end = start + (BLOCK_SIZE if i != seq.num_blocks - 1
                               else seq.last_block_num_tokens)
                slot_mapping.extend(range(start, end))
            blen = len(seq.block_table)
            if blen > max_bt:
                max_bt = blen

        block_tables = None
        if max_bt > 0:
            n = len(seqs)
            bt = np.full((n, max_bt), -1, dtype=np.int32)
            for i, seq in enumerate(seqs):
                if seq.block_table:
                    b = seq.block_table
                    bt[i, :len(b)] = b
            block_tables = torch.from_numpy(bt).pin_memory().cuda(non_blocking=True)

        set_context(
            True,
            torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_sq, max_sk,
            torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_tables=block_tables,
        )

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        if use_mrope:
            positions_t = torch.cat(mrope_pos_list, dim=1).to(torch.int64).pin_memory().cuda(non_blocking=True)
        else:
            positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        return input_ids_t, positions_t

    def prepare_decode(self, seqs):
        n = len(seqs)
        ids = np.empty(n, dtype=np.int64)
        pos = np.empty(n, dtype=np.int64)
        sm = np.empty(n, dtype=np.int32)
        cl = np.empty(n, dtype=np.int32)
        use_mrope = self.is_qwen_vl
        if use_mrope:
            mrope_pos = np.empty((3, n), dtype=np.int64)
        max_bt = 0
        for i, seq in enumerate(seqs):
            ids[i] = seq.last_token
            base_pos = len(seq) - 1
            if use_mrope:
                # For decode, all 3 dims get the same position = context_len + delta
                decode_pos = base_pos + seq.mrope_position_delta
                mrope_pos[:, i] = decode_pos
            else:
                pos[i] = base_pos
            cl[i] = len(seq)
            sm[i] = seq.block_table[-1] * BLOCK_SIZE + seq.last_block_num_tokens - 1
            blen = len(seq.block_table)
            if blen > max_bt:
                max_bt = blen
        bt = np.full((n, max_bt), -1, dtype=np.int32)
        for i, seq in enumerate(seqs):
            b = seq.block_table
            bt[i, :len(b)] = b
        set_context(
            False,
            slot_mapping=torch.from_numpy(sm).pin_memory().cuda(non_blocking=True),
            context_lens=torch.from_numpy(cl).pin_memory().cuda(non_blocking=True),
            block_tables=torch.from_numpy(bt).pin_memory().cuda(non_blocking=True),
        )
        if use_mrope:
            positions_t = torch.from_numpy(mrope_pos).pin_memory().cuda(non_blocking=True)
        else:
            positions_t = torch.from_numpy(pos).pin_memory().cuda(non_blocking=True)
        return (
            torch.from_numpy(ids).pin_memory().cuda(non_blocking=True),
            positions_t,
        )

    def prepare_mixed_batch(self, prefill_seqs, prefill_chunk_sizes, decode_seqs):
        """Prepare a unified mixed batch with full prefills and decode tokens.

        All attention reads from the paged KV cache via block_table.
        cu_seqlens_q/k cover all sequences (prefill + decode).
        """
        input_ids, positions = [], []
        slot_mapping = []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_sq, max_sk = 0, 0

        block_size = self.block_size
        all_seqs = list(prefill_seqs) + list(decode_seqs)
        max_bt = 0

        # Prefill sequences: q_len = prompt_len, k_len = prompt_len
        for seq, chunk_size in zip(prefill_seqs, prefill_chunk_sizes):
            sl = chunk_size
            input_ids.extend(seq.token_ids[:sl])
            positions.extend(range(sl))
            cu_seqlens_q.append(cu_seqlens_q[-1] + sl)
            cu_seqlens_k.append(cu_seqlens_k[-1] + sl)
            max_sq = max(sl, max_sq)
            max_sk = max(sl, max_sk)
            for i in range(seq.num_blocks):
                start = seq.block_table[i] * block_size
                end = start + (block_size if i != seq.num_blocks - 1
                               else seq.last_block_num_tokens)
                slot_mapping.extend(range(start, end))
            blen = len(seq.block_table)
            if blen > max_bt:
                max_bt = blen

        num_prefill_tokens = len(input_ids)

        # Decode sequences: q_len = 1, k_len = full context length
        for seq in decode_seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            cu_seqlens_q.append(cu_seqlens_q[-1] + 1)
            cu_seqlens_k.append(cu_seqlens_k[-1] + len(seq))
            max_sk = max(len(seq), max_sk)
            max_sq = max(1, max_sq)
            slot_mapping.append(
                seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
            )
            blen = len(seq.block_table)
            if blen > max_bt:
                max_bt = blen

        num_decode_tokens = len(decode_seqs)

        # Unified block table for all sequences
        n_all = len(all_seqs)
        bt = np.full((n_all, max_bt), -1, dtype=np.int32)
        for i, seq in enumerate(all_seqs):
            b = seq.block_table
            bt[i, :len(b)] = b

        set_mixed_context(
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max_sq,
            max_seqlen_k=max_sk,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            decode_context_lens=None,
            decode_block_tables=torch.from_numpy(bt).pin_memory().cuda(non_blocking=True),
        )
        return (
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
        )

    @torch.inference_mode()
    def run_model(self, input_ids, positions, is_prefill, inputs_embeds=None,
                  deepstack_embeds=None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs_list[-1]:
            if inputs_embeds is not None:
                return self.model.compute_logits(
                    self.model(input_ids, positions, inputs_embeds=inputs_embeds,
                               deepstack_embeds=deepstack_embeds)
                )
            return self.model.compute_logits(self.model(input_ids, positions))
        bs = input_ids.size(0)
        ctx = get_context()
        graph_bs = self._graph_bs_for_n[bs]
        gv = self.graph_vars
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        if bs < graph_bs:
            gv["slot_mapping"][bs:graph_bs].fill_(-1)
            gv["context_lens"][bs:graph_bs].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        bt = ctx.block_tables
        gv["block_tables"][:bs, :bt.size(1)] = bt
        self.graphs[graph_bs].replay()
        return self.model.compute_logits(gv["outputs"][:bs])

    @torch.inference_mode()
    def run_decode_greedy(self, seqs):
        """Fused decode path for greedy sampling with TP.
        Returns GPU tensor (rank 0) or list (TP=1).
        """
        decode_data = self._prepare_decode_arrays(seqs)
        return self.run_decode_greedy_fast(decode_data)

    @torch.inference_mode()
    def run_decode_greedy_fast(self, decode_data):
        """Fast decode: receives precomputed arrays instead of Sequence objects.
        
        Returns GPU tensor (rank 0) or None (other ranks).
        Does NOT call .tolist() -- caller is responsible for syncing.
        """
        n, ids_np, pos_np, sm_np, cl_np, bt_np = decode_data

        if self.enforce_eager:
            return self._run_decode_greedy_eager(n, ids_np, pos_np, sm_np, cl_np, bt_np)

        gv = self.graph_vars
        graph_bs = self._graph_bs_for_n[n]
        prev_n = getattr(self, '_prev_decode_n', -1)

        gv["input_ids"][:n].copy_(torch.from_numpy(ids_np), non_blocking=True)
        gv["positions"][:n].copy_(torch.from_numpy(pos_np), non_blocking=True)
        gv["slot_mapping"][:n].copy_(torch.from_numpy(sm_np), non_blocking=True)
        if n < graph_bs and n != prev_n:
            gv["slot_mapping"][n:graph_bs].fill_(-1)
            gv["context_lens"][n:graph_bs].zero_()
        gv["context_lens"][:n].copy_(torch.from_numpy(cl_np), non_blocking=True)
        gv["block_tables"][:n, :bt_np.shape[1]].copy_(
            torch.from_numpy(bt_np), non_blocking=True
        )
        self._prev_decode_n = n

        self.graphs[graph_bs].replay()
        return self._greedy_from_hidden(n)

    def _run_decode_greedy_eager(self, n, ids_np, pos_np, sm_np, cl_np, bt_np):
        """Eager decode path for greedy sampling with TP (no CUDA graphs)."""
        self._eager_input_ids[:n].copy_(torch.from_numpy(ids_np), non_blocking=True)
        self._eager_positions[:n].copy_(torch.from_numpy(pos_np), non_blocking=True)
        bt_cols = bt_np.shape[1]
        self._eager_slot_mapping[:n].copy_(torch.from_numpy(sm_np), non_blocking=True)
        self._eager_context_lens[:n].copy_(torch.from_numpy(cl_np), non_blocking=True)
        self._eager_block_tables[:n, :bt_cols].copy_(
            torch.from_numpy(bt_np), non_blocking=True)

        input_ids = self._eager_input_ids[:n]
        positions = self._eager_positions[:n]
        slot_mapping = self._eager_slot_mapping[:n]
        context_lens = self._eager_context_lens[:n]
        block_tables = self._eager_block_tables[:n, :bt_cols]

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        hidden = self.model(input_ids, positions)
        lm_head = self.model.lm_head
        logits = lm_head.linear_op(hidden, lm_head.weight).float()
        max_vals, max_idxs = logits.max(dim=-1)
        reset_context()

        if self.world_size > 1:
            info = self._greedy_info
            info[:n, 0] = max_vals
            info[:n, 1] = max_idxs.float()
            vocab_offset = lm_head.per_partition * self.rank
            info[:n, 1] += vocab_offset
            dist.all_gather(self._greedy_gathered, info)
            all_info = self._greedy_all_info
            torch.stack(self._greedy_gathered, out=all_info)
            best_rank = all_info[:, :n, 0].argmax(dim=0)
            return all_info[best_rank, self._greedy_arange[:n], 1].to(torch.int64)
        else:
            return max_idxs

    def _init_greedy_buffers(self):
        """Pre-allocate buffers for gather_greedy to avoid per-step allocation."""
        max_bs = MAX_NUM_SEQS
        dev = f"cuda:{self.rank}"
        self._greedy_info = torch.zeros(max_bs, 2, dtype=torch.float32, device=dev)
        self._greedy_gathered = [
            torch.zeros(max_bs, 2, dtype=torch.float32, device=dev)
            for _ in range(self.world_size)
        ]
        self._greedy_all_info = torch.zeros(self.world_size, max_bs, 2, dtype=torch.float32, device=dev)
        self._greedy_arange = torch.arange(max_bs, device=dev)

        max_num_blocks = (MAX_MODEL_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
        self._np_ids = np.empty(max_bs, dtype=np.int64)
        self._np_pos = np.empty(max_bs, dtype=np.int64)
        self._np_sm = np.empty(max_bs, dtype=np.int32)
        self._np_cl = np.empty(max_bs, dtype=np.int32)
        self._np_bt = np.full((max_bs, max_num_blocks), -1, dtype=np.int32)

        self._eager_input_ids = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self._eager_positions = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self._eager_slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=dev)
        self._eager_context_lens = torch.zeros(max_bs, dtype=torch.int32, device=dev)
        self._eager_block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=dev)

    def _greedy_from_hidden(self, n):
        """Use CUDA-graph-captured LM head + local argmax, then allgather.
        
        Returns GPU tensor of token IDs (rank 0) or None (other ranks).
        Caller must call .tolist() to sync.
        """
        gv = self.graph_vars
        local_max_vals = gv["lm_max_vals"][:n]
        local_max_idxs = gv["lm_max_idxs"][:n] + self.model.lm_head.vocab_start

        if self.world_size == 1:
            return local_max_idxs

        info = self._greedy_info[:n]
        info[:, 0] = local_max_vals
        info[:, 1] = local_max_idxs.float()

        gathered = [g[:n] for g in self._greedy_gathered]
        dist.all_gather(gathered, info)

        for i, g in enumerate(gathered):
            self._greedy_all_info[i, :n] = g
        all_vals = self._greedy_all_info[:, :n, 0]
        all_idxs = self._greedy_all_info[:, :n, 1].long()
        best_rank = all_vals.argmax(dim=0)
        token_ids = all_idxs[best_rank, self._greedy_arange[:n]]

        if self.rank == 0:
            return token_ids
        return None

    def _prepare_decode_arrays(self, seqs):
        """Precompute numpy arrays for decode - uses pre-allocated buffers."""
        n = len(seqs)
        ids_np = self._np_ids
        pos_np = self._np_pos
        sm_np = self._np_sm
        cl_np = self._np_cl
        max_bt = 0
        for i, seq in enumerate(seqs):
            ids_np[i] = seq.last_token
            if self.is_qwen_vl:
                pos_np[i] = len(seq) - 1 + seq.mrope_position_delta
            else:
                pos_np[i] = len(seq) - 1
            cl_np[i] = len(seq)
            sm_np[i] = seq.block_table[-1] * BLOCK_SIZE + seq.last_block_num_tokens - 1
            blen = len(seq.block_table)
            if blen > max_bt:
                max_bt = blen
        bt_np = self._np_bt
        for i, seq in enumerate(seqs):
            b = seq.block_table
            blen = len(b)
            bt_np[i, :blen] = b
            if blen < max_bt:
                bt_np[i, blen:max_bt] = -1
        return (n, ids_np[:n], pos_np[:n], sm_np[:n], cl_np[:n], bt_np[:n, :max_bt])

    @torch.inference_mode()
    def _run_mamba_seq(self, seq, is_prefill):
        if seq.state_slot is None:
            raise RuntimeError("Mamba sequence has no allocated state slot")
        slot_cache = self.mamba_state_manager.get_slot_cache(seq.state_slot)

        if is_prefill:
            input_ids = torch.tensor(
                seq.prompt_ids, dtype=torch.int64, device=f"cuda:{self.rank}",
            ).unsqueeze(0)
            cache_position = torch.arange(
                self.config.conv_kernel, dtype=torch.long, device=input_ids.device,
            )
        else:
            input_ids = torch.tensor(
                [[seq.last_token]], dtype=torch.int64, device=f"cuda:{self.rank}",
            )
            cache_position = torch.tensor(
                [seq.num_computed_tokens], dtype=torch.long, device=input_ids.device,
            )

        hidden_states = self.model(
            input_ids,
            cache_params=slot_cache,
            cache_position=cache_position,
        )
        logits = self.model.compute_logits(hidden_states[:, -1:, :])[:, -1, :]

        if is_prefill:
            seq.num_computed_tokens = seq.num_prompt_tokens
        else:
            seq.num_computed_tokens += 1
        return logits

    @torch.inference_mode()
    def _run_mamba_batch(self, seqs, is_prefill):
        outputs = [self._run_mamba_seq(seq, is_prefill) for seq in seqs]
        if not outputs:
            return None
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def _run_gla_seq(self, seq, is_prefill):
        if seq.state_slot is None:
            raise RuntimeError("GLA sequence has no allocated state slot")
        slot_cache = self.mamba_state_manager.get_slot_cache(seq.state_slot)

        if is_prefill:
            input_ids = torch.tensor(
                seq.prompt_ids, dtype=torch.int64, device=f"cuda:{self.rank}",
            ).unsqueeze(0)
        else:
            input_ids = torch.tensor(
                [[seq.last_token]], dtype=torch.int64, device=f"cuda:{self.rank}",
            )

        hidden_states = self.model(
            input_ids,
            past_key_values=slot_cache,
            use_cache=True,
        )
        logits = self.model.compute_logits(hidden_states[:, -1:, :])[:, -1, :]

        if is_prefill:
            seq.num_computed_tokens = seq.num_prompt_tokens
        else:
            seq.num_computed_tokens += 1
        return logits

    @torch.inference_mode()
    def _run_gla_batch(self, seqs, is_prefill):
        outputs = [self._run_gla_seq(seq, is_prefill) for seq in seqs]
        if not outputs:
            return None
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def _run_rwkv7_seq(self, seq, is_prefill):
        if seq.state_slot is None:
            raise RuntimeError("RWKV7 sequence has no allocated state slot")
        slot_cache = self.mamba_state_manager.get_slot_cache(seq.state_slot)

        if is_prefill:
            input_ids = torch.tensor(
                seq.prompt_ids,
                dtype=torch.int64,
                device=f"cuda:{self.rank}",
            ).unsqueeze(0)
        else:
            input_ids = torch.tensor(
                [[seq.last_token]],
                dtype=torch.int64,
                device=f"cuda:{self.rank}",
            )

        hidden_states = self.model(
            input_ids,
            past_key_values=slot_cache,
            use_cache=True,
        )
        logits = self.model.compute_logits(hidden_states[:, -1:, :])[:, -1, :]

        if is_prefill:
            seq.num_computed_tokens = seq.num_prompt_tokens
        else:
            seq.num_computed_tokens += 1
        return logits

    @torch.inference_mode()
    def _run_rwkv7_batch(self, seqs, is_prefill):
        outputs = [self._run_rwkv7_seq(seq, is_prefill) for seq in seqs]
        if not outputs:
            return None
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def _run_kimi_linear_seq(self, seq, is_prefill):
        if seq.state_slot is None:
            raise RuntimeError("Kimi-Linear sequence has no allocated state slot")
        slot_cache = self.mamba_state_manager.get_slot_cache(seq.state_slot)

        if is_prefill:
            # Use token_ids (preserved across pickle) instead of prompt_ids
            # (stripped by __getstate__ for workers)
            input_ids = torch.tensor(
                seq.token_ids, dtype=torch.int64, device=f"cuda:{self.rank}",
            ).unsqueeze(0)
        else:
            input_ids = torch.tensor(
                [[seq.last_token]], dtype=torch.int64, device=f"cuda:{self.rank}",
            )

        T = input_ids.shape[1]
        positions = torch.arange(
            seq.num_computed_tokens,
            seq.num_computed_tokens + T,
            dtype=torch.int64,
            device=input_ids.device,
        )

        hidden_states = self.model(
            input_ids,
            positions=positions,
            past_key_values=slot_cache,
            use_cache=True,
        )
        logits = self.model.compute_logits(hidden_states[:, -1:, :])[:, -1, :]

        if is_prefill:
            seq.num_computed_tokens = T
        else:
            seq.num_computed_tokens += 1
        return logits

    @torch.inference_mode()
    def _run_kimi_linear_batch(self, seqs, is_prefill):
        outputs = [self._run_kimi_linear_seq(seq, is_prefill) for seq in seqs]
        if not outputs:
            return None
        return torch.cat(outputs, dim=0)

    def _run_qwen3_next_seq(self, seq, is_prefill):
        if seq.state_slot is None:
            raise RuntimeError("Qwen3-Next sequence has no allocated state slot")
        slot_cache = self.mamba_state_manager.get_slot_cache(seq.state_slot)

        if is_prefill:
            input_ids = torch.tensor(
                seq.token_ids, dtype=torch.int64, device=f"cuda:{self.rank}",
            ).unsqueeze(0)
        else:
            input_ids = torch.tensor(
                [[seq.last_token]], dtype=torch.int64, device=f"cuda:{self.rank}",
            )

        T = input_ids.shape[1]
        positions = torch.arange(
            seq.num_computed_tokens,
            seq.num_computed_tokens + T,
            dtype=torch.int64,
            device=input_ids.device,
        )

        hidden_states = self.model(
            input_ids,
            positions=positions,
            layer_states=slot_cache,
        )
        logits = self.model.compute_logits(hidden_states[:, -1:, :])[:, -1, :]

        if is_prefill:
            seq.num_computed_tokens = T
        else:
            seq.num_computed_tokens += 1
        return logits

    @torch.inference_mode()
    def _run_qwen3_next_batch(self, seqs, is_prefill):
        outputs = [self._run_qwen3_next_seq(seq, is_prefill) for seq in seqs]
        if not outputs:
            return None
        return torch.cat(outputs, dim=0)

    def _write_decode_shm(self, n, ids_np, pos_np, sm_np, cl_np, bt_np):
        """Write decode arrays directly into SHM with binary layout.
        
        Layout: [n(2)][max_bt(2)][ids(n*8)][pos(n*8)][sm(n*4)][cl(n*4)][bt(n*max_bt*4)]
        """
        max_bt = bt_np.shape[1]
        buf = self.shm.buf
        buf[0:2] = n.to_bytes(2, "little")
        buf[2:4] = max_bt.to_bytes(2, "little")
        off = 4
        for arr in (ids_np, pos_np, sm_np, cl_np, bt_np):
            nb = arr.nbytes
            buf[off:off+nb] = arr.tobytes()
            off += nb

    def _loop_decode_greedy(self):
        """Worker fast path: read decode arrays from SHM without pickle."""
        buf = self.shm.buf
        n = int.from_bytes(buf[0:2], "little")
        max_bt = int.from_bytes(buf[2:4], "little")
        off = 4
        ids_np = np.frombuffer(buf, dtype=np.int64, count=n, offset=off).copy(); off += n * 8
        pos_np = np.frombuffer(buf, dtype=np.int64, count=n, offset=off).copy(); off += n * 8
        sm_np = np.frombuffer(buf, dtype=np.int32, count=n, offset=off).copy(); off += n * 4
        cl_np = np.frombuffer(buf, dtype=np.int32, count=n, offset=off).copy(); off += n * 4
        bt_np = np.frombuffer(buf, dtype=np.int32, count=n*max_bt, offset=off).copy().reshape(n, max_bt)
        self.run_decode_greedy_fast((n, ids_np, pos_np, sm_np, cl_np, bt_np))

    def call_decode_greedy(self, seqs):
        """Optimized call for greedy decode: uses SHM spin-wait signaling.
        
        Returns GPU tensor of token IDs (doesn't sync).
        Caller must call .tolist() to get Python list.
        """
        if self.world_size > 1 and self.rank == 0:
            if _PROFILE:
                _t0 = time.perf_counter()
            decode_data = self._prepare_decode_arrays(seqs)
            self._write_decode_shm(*decode_data)
            if _PROFILE:
                _t1 = time.perf_counter()
            self.shm.buf[self._SHM_FLAG_OFFSET] = 1  # mark as decode_greedy
            self._signal_workers()
            if _PROFILE:
                _t2 = time.perf_counter()
            result = self.run_decode_greedy_fast(decode_data)
            if _PROFILE:
                torch.cuda.synchronize()
                _t3 = time.perf_counter()
                pd = getattr(self, '_call_profile', None)
                if pd is None:
                    pd = {"prepare": 0.0, "signal": 0.0, "gpu_exec": 0.0, "n_calls": 0}
                    self._call_profile = pd
                pd["prepare"] += _t1 - _t0
                pd["signal"] += _t2 - _t1
                pd["gpu_exec"] += _t3 - _t2
                pd["n_calls"] += 1
            return result
        return self.run_decode_greedy(seqs)

    def run(self, seqs, is_prefill):
        if self.model_family == "mamba":
            model_type = getattr(self.config, "model_type", "")
            if model_type == "gla":
                return self._run_gla_batch(seqs, is_prefill)
            if model_type == "rwkv7":
                return self._run_rwkv7_batch(seqs, is_prefill)
            if model_type == "kimi_linear":
                return self._run_kimi_linear_batch(seqs, is_prefill)
            if model_type == "qwen3_next":
                return self._run_qwen3_next_batch(seqs, is_prefill)
            return self._run_mamba_batch(seqs, is_prefill)
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill
            else self.prepare_decode(seqs)
        )
        result = self.run_model(input_ids, positions, is_prefill)
        reset_context()
        return result

    def run_mixed(self, prefill_seqs, prefill_chunk_sizes, decode_seqs):
        input_ids, positions = self.prepare_mixed_batch(
            prefill_seqs, prefill_chunk_sizes, decode_seqs,
        )
        result = self.run_model(input_ids, positions, True)
        reset_context()
        return result

    @torch.inference_mode()
    def capture_cudagraph(self):
        from contextlib import nullcontext
        max_bs = MAX_NUM_SEQS
        max_num_blocks = (MAX_MODEL_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)

        self.graph_bs_list = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        outputs = torch.zeros(max_bs, self.config.hidden_size)

        lm_head = self.model.lm_head
        vocab_per_rank = lm_head.per_partition
        lm_logits = torch.zeros(max_bs, vocab_per_rank)
        lm_max_vals = torch.zeros(max_bs)
        lm_max_idxs = torch.zeros(max_bs, dtype=torch.int64)

        ar_ctx = self.custom_ar.capture() if self.custom_ar is not None else nullcontext()
        with ar_ctx:
            for bs in reversed(self.graph_bs_list):
                graph = torch.cuda.CUDAGraph()
                set_context(
                    False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                )
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                lm_logits[:bs] = lm_head.linear_op(outputs[:bs], lm_head.weight).float()
                lm_max_vals[:bs], lm_max_idxs[:bs] = lm_logits[:bs].max(dim=-1)

                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                    lm_logits[:bs] = lm_head.linear_op(outputs[:bs], lm_head.weight).float()
                    lm_max_vals[:bs], lm_max_idxs[:bs] = lm_logits[:bs].max(dim=-1)

                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
                reset_context()

        self.graph_vars = dict(
            input_ids=input_ids, positions=positions,
            slot_mapping=slot_mapping, context_lens=context_lens,
            block_tables=block_tables, outputs=outputs,
            lm_logits=lm_logits, lm_max_vals=lm_max_vals,
            lm_max_idxs=lm_max_idxs,
        )

        # Pre-compute lookup table: _graph_bs_for_n[n] = smallest graph_bs >= n
        self._graph_bs_for_n = [0] * (max_bs + 1)
        for n in range(max_bs + 1):
            self._graph_bs_for_n[n] = next(x for x in self.graph_bs_list if x >= n)


# ---------------------------------------------------------------------------
# LlamaEngine — only runs on rank 0
# ---------------------------------------------------------------------------
class LlamaEngine:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
    ):
        self.model_name = model_name
        self.seed = seed
        self._set_seeds(seed)
        model_type = getattr(AutoConfig.from_pretrained(model_name, trust_remote_code=True), "model_type", "")
        self.model_family = (
            "mamba" if model_type in {"mamba", "mamba2", "gla", "rwkv7", "kimi_linear", "qwen3_next"}
            else "attention"
        )
        self.model_type = model_type
        if self.model_family == "mamba":
            if tensor_parallel_size != 1 and model_type not in {"kimi_linear", "qwen3_next"}:
                raise ValueError("Recurrent-family currently supports tensor_parallel_size=1 only")
            if not enforce_eager:
                enforce_eager = True

        # Unique shared memory name to avoid collisions
        shm_name = f"sllama_{uuid.uuid4().hex[:8]}"

        # Launch non-rank-0 workers
        self.workers = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, tensor_parallel_size):
            event = ctx.Event()
            p = ctx.Process(
                target=ModelRunner,
                args=(model_name, i, tensor_parallel_size, dtype,
                      enforce_eager, event, shm_name),
            )
            p.start()
            self.workers.append(p)
            self.events.append(event)

        # Rank 0 model runner (events is a list for rank 0)
        self.model_runner = ModelRunner(
            model_name, 0, tensor_parallel_size, dtype,
            enforce_eager, self.events, shm_name,
        )
        self.block_manager = (
            BlockManager(self.model_runner.num_blocks)
            if self.model_family == "attention" else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.is_qwen_vl = self.model_runner.is_qwen_vl
        self.processor = None
        if self.is_qwen_vl:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name)

        atexit.register(self._cleanup)

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _cleanup(self):
        if hasattr(self, "model_runner"):
            try:
                self.model_runner.call("exit")
            except Exception:
                pass
            del self.model_runner
            for p in self.workers:
                p.join(timeout=10)
            torch.cuda.empty_cache()

    def _sample_greedy(self, logits):
        return logits.argmax(dim=-1).tolist()

    def _sample(self, logits, params):
        if logits is None:
            return []
        if params.temperature == 0.0:
            return self._sample_greedy(logits)
        logits = logits / params.temperature
        if params.top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(torch.softmax(sl, -1), -1)
            mask = cp - torch.softmax(sl, -1) >= params.top_p
            sl[mask] = float("-inf")
            logits = logits.scatter(1, si, sl)
        probs = torch.softmax(logits, -1)
        return torch.multinomial(probs, 1).squeeze(-1).tolist()

    def _preprocess_multimodal(self, prompt, images=None, videos=None):
        """Preprocess a multimodal prompt with images/videos.

        Returns (token_ids, pixel_values, image_grid_thw, video_pixel_values,
                 video_grid_thw) where pixel values are already processed.
        """
        messages = [{"role": "user", "content": []}]
        if images:
            for img in images:
                messages[0]["content"].append({"type": "image", "image": img})
        if videos:
            for vid in videos:
                messages[0]["content"].append({"type": "video", "video": vid})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=images, videos=videos,
            return_tensors="pt", padding=True,
        )
        token_ids = inputs["input_ids"][0].tolist()
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        video_pixel_values = inputs.get("pixel_values_videos", None)
        video_grid_thw = inputs.get("video_grid_thw", None)

        return (token_ids, pixel_values, image_grid_thw,
                video_pixel_values, video_grid_thw)

    @torch.inference_mode()
    def _run_vision_encoder(self, seqs):
        """Run vision encoder for sequences with multimodal data and merge embeddings.

        Returns (inputs_embeds, deepstack_embeds) where deepstack_embeds is a list
        of tensors for Qwen3-VL DeepStack, or None for Qwen2-VL.
        """
        model = self.model_runner.model
        all_inputs_embeds = []
        has_deepstack = hasattr(model.visual, 'deepstack_merger_list')
        all_deepstack = [] if has_deepstack else None

        for seq in seqs:
            token_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cuda")
            text_embeds = model.get_input_embeddings()(token_ids)
            seq_deepstack = [] if has_deepstack else None

            if seq.pixel_values is not None:
                pixel_values = seq.pixel_values.cuda()
                grid_thw = seq.image_grid_thw
                vis_out = model.visual(pixel_values, grid_thw=grid_thw)

                if has_deepstack:
                    image_embeds, ds_features = vis_out
                else:
                    image_embeds = vis_out
                    ds_features = []

                merge_size = model.config.vision.spatial_merge_size
                sizes = []
                for thw in grid_thw:
                    t, h, w = thw
                    sizes.append(t * (h // merge_size) * (w // merge_size))

                mask = token_ids == QWEN_IMAGE_PAD_ID
                if mask.any():
                    text_embeds[mask] = image_embeds.to(text_embeds.dtype)

                if has_deepstack and ds_features:
                    for ds_feat in ds_features:
                        ds_expanded = torch.zeros_like(text_embeds)
                        if mask.any():
                            ds_expanded[mask] = ds_feat.to(text_embeds.dtype)
                        seq_deepstack.append(ds_expanded)

            if seq.video_pixel_values is not None:
                video_pv = seq.video_pixel_values.cuda()
                grid_thw = seq.video_grid_thw
                vis_out = model.visual(video_pv, grid_thw=grid_thw)

                if has_deepstack:
                    video_embeds, ds_features = vis_out
                else:
                    video_embeds = vis_out
                    ds_features = []

                mask = token_ids == QWEN_VIDEO_PAD_ID
                if mask.any():
                    text_embeds[mask] = video_embeds.to(text_embeds.dtype)

                if has_deepstack and ds_features:
                    for i, ds_feat in enumerate(ds_features):
                        ds_expanded = torch.zeros_like(text_embeds)
                        if mask.any():
                            ds_expanded[mask] = ds_feat.to(text_embeds.dtype)
                        if i < len(seq_deepstack):
                            seq_deepstack[i] = seq_deepstack[i] + ds_expanded
                        else:
                            seq_deepstack.append(ds_expanded)

            all_inputs_embeds.append(text_embeds)
            if has_deepstack:
                all_deepstack.append(seq_deepstack)

        inputs_embeds = torch.cat(all_inputs_embeds, dim=0)

        if has_deepstack and all_deepstack:
            num_levels = max(len(ds) for ds in all_deepstack)
            deepstack_embeds = []
            for level in range(num_levels):
                level_parts = []
                for ds in all_deepstack:
                    if level < len(ds):
                        level_parts.append(ds[level])
                    else:
                        level_parts.append(torch.zeros_like(all_inputs_embeds[0]))
                deepstack_embeds.append(torch.cat(level_parts, dim=0))
            return inputs_embeds, deepstack_embeds

        return inputs_embeds, None

    @torch.inference_mode()
    def generate(self, prompts, sampling_params, collect_logits: bool = False,
                 images=None, videos=None):
        """Generate completions for a batch of prompts."""
        if isinstance(sampling_params, list):
            sp_list = sampling_params
        else:
            sp_list = [sampling_params] * len(prompts)

        seed = sp_list[0].seed
        if seed is not None:
            self._set_seeds(seed)

        eos = self.tokenizer.eos_token_id
        waiting = deque()
        running = deque()

        seq_logits: dict[int, list[torch.Tensor]] = {}

        # Handle multimodal inputs
        if images is None:
            images = [None] * len(prompts)
        if videos is None:
            videos = [None] * len(prompts)

        for i, (prompt, sp) in enumerate(zip(prompts, sp_list)):
            img = images[i] if i < len(images) else None
            vid = videos[i] if i < len(videos) else None

            if self.is_qwen_vl and (img is not None or vid is not None):
                (ids, pixel_values, image_grid_thw,
                 video_pv, video_grid_thw) = self._preprocess_multimodal(
                    prompt, images=img, videos=vid,
                )
                seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
                seq.pixel_values = pixel_values
                seq.image_grid_thw = image_grid_thw.tolist() if image_grid_thw is not None else None
                seq.video_pixel_values = video_pv
                seq.video_grid_thw = video_grid_thw.tolist() if video_grid_thw is not None else None

                # Compute M-RoPE positions
                model = self.model_runner.model
                merge_size = model.config.vision.spatial_merge_size
                image_offsets = []
                video_offsets = []
                img_idx = 0
                vid_idx = 0
                i_tok = 0
                while i_tok < len(ids):
                    tid = ids[i_tok]
                    if tid == QWEN_IMAGE_PAD_ID and seq.image_grid_thw and img_idx < len(seq.image_grid_thw):
                        image_offsets.append(i_tok)
                        t, h, w = seq.image_grid_thw[img_idx]
                        num_tokens = t * (h // merge_size) * (w // merge_size)
                        i_tok += num_tokens
                        img_idx += 1
                    elif tid == QWEN_VIDEO_PAD_ID and seq.video_grid_thw and vid_idx < len(seq.video_grid_thw):
                        video_offsets.append(i_tok)
                        t, h, w = seq.video_grid_thw[vid_idx]
                        num_tokens = t * (h // merge_size) * (w // merge_size)
                        i_tok += num_tokens
                        vid_idx += 1
                    else:
                        i_tok += 1

                mrope_positions, delta = model.get_mrope_input_positions(
                    ids,
                    image_grid_thw=seq.image_grid_thw,
                    video_grid_thw=seq.video_grid_thw,
                    image_offsets=image_offsets if image_offsets else None,
                    video_offsets=video_offsets if video_offsets else None,
                )
                seq.mrope_positions = mrope_positions
                seq.mrope_position_delta = delta
            elif self.is_qwen_vl:
                ids = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
                seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
                # Text-only with M-RoPE: all 3 dims same
                seq.mrope_positions = torch.arange(len(ids), dtype=torch.int64).unsqueeze(0).expand(3, -1)
                seq.mrope_position_delta = 0
            else:
                ids = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
                seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)

            waiting.append(seq)
            if collect_logits:
                seq_logits[id(seq)] = []

        all_seqs = list(waiting)

        if self.model_family == "mamba":
            while waiting or running:
                prefill_seqs = []
                while (
                    waiting
                    and len(prefill_seqs) < MAX_NUM_SEQS
                    and self.model_runner.can_allocate_mamba_state()
                ):
                    seq = waiting.popleft()
                    self.model_runner.allocate_mamba_state(seq)
                    seq.status = SeqStatus.RUNNING
                    running.append(seq)
                    prefill_seqs.append(seq)

                if prefill_seqs:
                    logits = self.model_runner.call("run", prefill_seqs, True)
                    if logits is not None:
                        if collect_logits:
                            for i, seq in enumerate(prefill_seqs):
                                seq_logits[id(seq)].append(logits[i:i+1].cpu())
                        token_ids = self._sample(logits, sp_list[0])
                        finished_set = set()
                        for seq, tid in zip(prefill_seqs, token_ids):
                            seq.append_token(tid)
                            done = len(seq.generated_ids) >= seq.max_tokens
                            if not seq.ignore_eos:
                                done = done or tid == eos
                            if done:
                                seq.status = SeqStatus.FINISHED
                                finished_set.add(id(seq))
                                self.model_runner.deallocate_mamba_state(seq)
                        if finished_set:
                            running = deque(s for s in running if id(s) not in finished_set)

                if not running:
                    continue

                decode_seqs = list(running)
                result = self.model_runner.call("run", decode_seqs, False)
                if result is not None:
                    if collect_logits:
                        for i, seq in enumerate(decode_seqs):
                            seq_logits[id(seq)].append(result[i:i+1].cpu())
                    token_ids = self._sample(result, sp_list[0])
                    finished_set = set()
                    for seq, tid in zip(decode_seqs, token_ids):
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            seq.status = SeqStatus.FINISHED
                            finished_set.add(id(seq))
                            self.model_runner.deallocate_mamba_state(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)

            return [
                GenerationOutput(
                    prompt=(prompts[i] if isinstance(prompts[i], str) else ""),
                    generated_text=self.tokenizer.decode(
                        all_seqs[i].generated_ids, skip_special_tokens=True,
                    ),
                    token_ids=all_seqs[i].generated_ids,
                    logits_history=(
                        seq_logits.get(id(all_seqs[i])) if collect_logits else None
                    ),
                )
                for i in range(len(prompts))
            ]

        use_greedy = (sp_list[0].temperature == 0.0
                      and not collect_logits)
        block_size = BLOCK_SIZE

        profile = _PROFILE
        if profile:
            _pf_time = 0.0
            _pf_steps = 0
            _pf_tokens = 0
            _dc_time = 0.0
            _dc_steps = 0
            _dc_tokens = 0
            _dc_sched_time = 0.0
            _dc_call_time = 0.0
            _dc_tolist_time = 0.0
            _dc_post_time = 0.0
            _dc_bs_counts = []

        while waiting or running:
            # --- Prefill one batch (if any waiting) ---
            prefill_seqs = []
            num_batched_tokens = 0
            while waiting:
                seq = waiting[0]
                seq_len = len(seq)
                if num_batched_tokens + seq_len > MAX_NUM_BATCHED_TOKENS:
                    break
                if len(prefill_seqs) >= MAX_NUM_SEQS:
                    break
                if not self.block_manager.can_allocate(seq):
                    break
                waiting.popleft()
                self.block_manager.allocate(seq)
                seq.status = SeqStatus.RUNNING
                running.append(seq)
                prefill_seqs.append(seq)
                num_batched_tokens += seq_len

            if prefill_seqs:
                if profile:
                    _t0 = time.perf_counter()
                # Check if any sequences have multimodal data
                has_mm = any(s.pixel_values is not None or s.video_pixel_values is not None
                             for s in prefill_seqs)
                if has_mm:
                    inputs_embeds, deepstack_embeds = self._run_vision_encoder(prefill_seqs)
                    input_ids_t, positions_t = self.model_runner.prepare_prefill(prefill_seqs)
                    logits = self.model_runner.run_model(
                        input_ids_t, positions_t, True,
                        inputs_embeds=inputs_embeds,
                        deepstack_embeds=deepstack_embeds,
                    )
                    reset_context()
                else:
                    logits = self.model_runner.call("run", prefill_seqs, True)
                if logits is not None:
                    if collect_logits:
                        for i, seq in enumerate(prefill_seqs):
                            seq_logits[id(seq)].append(logits[i:i+1].cpu())
                    token_ids = self._sample(logits, sp_list[0])
                    for seq, tid in zip(prefill_seqs, token_ids):
                        seq.num_computed_tokens = len(seq)
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            seq.status = SeqStatus.FINISHED
                            running.remove(seq)
                            self.block_manager.deallocate(seq)
                if profile:
                    _pf_time += time.perf_counter() - _t0
                    _pf_steps += 1
                    _pf_tokens += num_batched_tokens

            # --- Decode all running sequences (CUDA graph path) ---
            if not running:
                continue

            if profile:
                _t_sched = time.perf_counter()

            bm = self.block_manager
            free = bm.free_block_ids
            decode_seqs = []
            temp = deque()
            while running and len(decode_seqs) < MAX_NUM_SEQS:
                seq = running.popleft()
                if len(seq) % block_size == 1 and not free:
                    break
                if len(seq) % block_size == 1:
                    seq.block_table.append(free.popleft())
                decode_seqs.append(seq)
                temp.append(seq)
            running.extendleft(reversed(temp))

            if not decode_seqs:
                if not waiting:
                    break
                continue

            if profile:
                _dc_sched_time += time.perf_counter() - _t_sched
                _dc_bs_counts.append(len(decode_seqs))
                _t_call = time.perf_counter()

            if use_greedy:
                gpu_result = self.model_runner.call_decode_greedy(decode_seqs)
                if profile:
                    _dc_call_time += time.perf_counter() - _t_call
                    _t_tolist = time.perf_counter()
                if gpu_result is not None:
                    token_ids = gpu_result.tolist()
                    if profile:
                        _dc_tolist_time += time.perf_counter() - _t_tolist
                        _t_post = time.perf_counter()
                    finished_set = set()
                    for seq, tid in zip(decode_seqs, token_ids):
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            seq.status = SeqStatus.FINISHED
                            finished_set.add(id(seq))
                            bm.deallocate(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)
                    if profile:
                        _dc_post_time += time.perf_counter() - _t_post
                elif profile:
                    _dc_tolist_time += time.perf_counter() - _t_tolist
                    _dc_post_time += 0.0
            else:
                result = self.model_runner.call("run", decode_seqs, False)
                if profile:
                    _dc_call_time += time.perf_counter() - _t_call
                    _t_tolist = time.perf_counter()
                    _dc_tolist_time += 0.0
                    _t_post = time.perf_counter()
                if result is not None:
                    if collect_logits:
                        for i, seq in enumerate(decode_seqs):
                            seq_logits[id(seq)].append(result[i:i+1].cpu())
                    token_ids = self._sample(result, sp_list[0])
                    finished_set = set()
                    for seq, tid in zip(decode_seqs, token_ids):
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            seq.status = SeqStatus.FINISHED
                            finished_set.add(id(seq))
                            bm.deallocate(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)
                if profile:
                    _dc_post_time += time.perf_counter() - _t_post

            if profile:
                _dc_steps += 1
                _dc_tokens += len(decode_seqs)

        if profile:
            self._profile_data = {
                "prefill_time": _pf_time,
                "prefill_steps": _pf_steps,
                "prefill_tokens": _pf_tokens,
                "decode_time": _dc_time,
                "decode_steps": _dc_steps,
                "decode_tokens": _dc_tokens,
                "decode_sched_time": _dc_sched_time,
                "decode_call_time": _dc_call_time,
                "decode_tolist_time": _dc_tolist_time,
                "decode_post_time": _dc_post_time,
                "decode_bs_counts": _dc_bs_counts,
            }
            self._profile_data["decode_time"] = (
                _dc_sched_time + _dc_call_time + _dc_tolist_time + _dc_post_time
            )
            cp = getattr(self.model_runner, '_call_profile', None)
            if cp and cp["n_calls"] > 0:
                self._profile_data["call_detail"] = {
                    "prepare_ms": cp["prepare"] / cp["n_calls"] * 1000,
                    "signal_ms": cp["signal"] / cp["n_calls"] * 1000,
                    "gpu_exec_ms": cp["gpu_exec"] / cp["n_calls"] * 1000,
                    "n_calls": cp["n_calls"],
                }

        # Return in original order
        return [
            GenerationOutput(
                prompt=(prompts[i] if isinstance(prompts[i], str) else ""),
                generated_text=self.tokenizer.decode(
                    all_seqs[i].generated_ids, skip_special_tokens=True,
                ),
                token_ids=all_seqs[i].generated_ids,
                logits_history=(
                    seq_logits.get(id(all_seqs[i])) if collect_logits else None
                ),
            )
            for i in range(len(prompts))
        ]
