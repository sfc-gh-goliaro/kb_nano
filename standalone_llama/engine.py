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
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from .ops import get_context, reset_context, set_context
from .weight_loader import load_model

BLOCK_SIZE = 256
NCCL_PORT = 29501


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = None


@dataclass
class GenerationOutput:
    prompt: str
    generated_text: str
    token_ids: list[int] = field(default_factory=list)


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# ---------------------------------------------------------------------------
# Sequence — must be picklable for shared memory transfer
# ---------------------------------------------------------------------------
class Sequence:
    _next_id = 0

    def __init__(self, prompt_ids: list[int], max_tokens: int = 512):
        self.seq_id = Sequence._next_id
        Sequence._next_id += 1
        self.prompt_ids = list(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.generated_ids: list[int] = []
        self.max_tokens = max_tokens
        self.block_table: list[int] = []
        self.status = SeqStatus.WAITING

    def __len__(self):
        return len(self.token_ids)

    @property
    def num_blocks(self):
        return (len(self) + BLOCK_SIZE - 1) // BLOCK_SIZE

    @property
    def last_block_num_tokens(self):
        r = len(self) % BLOCK_SIZE
        return r if r else BLOCK_SIZE

    @property
    def last_token(self):
        return self.token_ids[-1]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.generated_ids.append(token_id)

    def __getstate__(self):
        """Minimal pickling for shared memory transfer to non-rank-0 workers."""
        return (len(self.token_ids), len(self.prompt_ids), self.block_table,
                self.token_ids if not self.generated_ids else self.last_token)

    def __setstate__(self, state):
        num_tokens, num_prompt, self.block_table = state[:-1]
        if isinstance(state[-1], list):
            self.token_ids = state[-1]
        else:
            self.token_ids = [0] * (num_tokens - 1) + [state[-1]]
        self.prompt_ids = self.token_ids[:num_prompt]
        self.generated_ids = self.token_ids[num_prompt:]
        self.num_tokens_saved = num_tokens


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

        dist.init_process_group(
            "nccl", f"tcp://localhost:{NCCL_PORT}",
            world_size=world_size, rank=rank,
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")

        self.model, self.config = load_model(
            model_name, torch.device(f"cuda:{rank}"), dtype,
        )
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # TP shared memory setup
        if world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=shm_name)
                self.loop()  # Non-rank-0 blocks here forever

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """Worker loop for non-rank-0 processes."""
        while True:
            self.event.wait()
            n = int.from_bytes(self.shm.buf[0:4], "little")
            method_name, *args = pickle.loads(self.shm.buf[4:n+4])
            self.event.clear()
            getattr(self, method_name)(*args)
            if method_name == "exit":
                break

    def call(self, method_name, *args):
        """Called by rank 0 to execute method on ALL ranks."""
        if self.world_size > 1 and self.rank == 0:
            data = pickle.dumps([method_name, *args])
            n = len(data)
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4:n+4] = data
            for ev in self.event:
                ev.set()
        return getattr(self, method_name)(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seqs = [Sequence([0] * 256)]
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

    def prepare_prefill(self, seqs):
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_sq, max_sk = 0, 0
        slot_mapping = []
        for seq in seqs:
            sl = len(seq)
            input_ids.extend(seq.token_ids)
            positions.extend(range(sl))
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

        set_context(
            True,
            torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_sq, max_sk,
            torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
        )
        return (
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
        )

    def prepare_decode(self, seqs):
        input_ids, positions, slot_mapping, context_lens = [], [], [], []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * BLOCK_SIZE + seq.last_block_num_tokens - 1
            )
        max_bt = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_bt - len(seq.block_table))
            for seq in seqs
        ]
        set_context(
            False,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
        )
        return (
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
        )

    @torch.inference_mode()
    def run_model(self, input_ids, positions, is_prefill):
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs_list[-1]:
            return self.model.compute_logits(self.model(input_ids, positions))
        bs = input_ids.size(0)
        ctx = get_context()
        graph_bs = next(x for x in self.graph_bs_list if x >= bs)
        gv = self.graph_vars
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = ctx.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = ctx.context_lens
        bt = ctx.block_tables
        gv["block_tables"][:bs, :bt.size(1)] = bt
        self.graphs[graph_bs].replay()
        return self.model.compute_logits(gv["outputs"][:bs])

    def run(self, seqs, is_prefill):
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill
            else self.prepare_decode(seqs)
        )
        logits = self.run_model(input_ids, positions, is_prefill)
        reset_context()
        return logits

    @torch.inference_mode()
    def capture_cudagraph(self):
        max_bs = 64
        max_num_blocks = 256
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, self.config.hidden_size)

        self.graph_bs_list = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs_list):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False, slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs], block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids, positions=positions,
            slot_mapping=slot_mapping, context_lens=context_lens,
            block_tables=block_tables, outputs=outputs,
        )


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
        self.seed = seed
        self._set_seeds(seed)

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
        self.block_manager = BlockManager(self.model_runner.num_blocks)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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

    @torch.inference_mode()
    def generate(self, prompts, sampling_params):
        if sampling_params.seed is not None:
            self._set_seeds(sampling_params.seed)

        eos = self.tokenizer.eos_token_id
        waiting = deque()
        running = deque()

        for prompt in prompts:
            ids = self.tokenizer.encode(prompt)
            seq = Sequence(ids, max_tokens=sampling_params.max_tokens)
            waiting.append(seq)

        all_seqs = list(waiting)

        while waiting or running:
            # --- Prefill ---
            prefill_seqs = []
            while waiting:
                seq = waiting[0]
                if not self.block_manager.can_allocate(seq):
                    break
                waiting.popleft()
                self.block_manager.allocate(seq)
                seq.status = SeqStatus.RUNNING
                running.append(seq)
                prefill_seqs.append(seq)

            if prefill_seqs:
                logits = self.model_runner.call("run", prefill_seqs, True)
                if logits is not None:
                    token_ids = self._sample(logits, sampling_params)
                    for seq, tid in zip(prefill_seqs, token_ids):
                        seq.append_token(tid)
                        if tid == eos or len(seq.generated_ids) >= seq.max_tokens:
                            seq.status = SeqStatus.FINISHED
                            running.remove(seq)
                            self.block_manager.deallocate(seq)
                continue

            # --- Decode ---
            decode_seqs = []
            temp = deque()
            while running:
                seq = running.popleft()
                if not self.block_manager.can_append(seq):
                    break
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                temp.append(seq)
            running.extendleft(reversed(temp))

            if not decode_seqs:
                break

            logits = self.model_runner.call("run", decode_seqs, False)
            if logits is not None:
                token_ids = self._sample(logits, sampling_params)
                for seq, tid in zip(decode_seqs, token_ids):
                    seq.append_token(tid)
                    if tid == eos or len(seq.generated_ids) >= seq.max_tokens:
                        seq.status = SeqStatus.FINISHED
                        running.remove(seq)
                        self.block_manager.deallocate(seq)

        # Return in original order
        return [
            GenerationOutput(
                prompt=prompts[i],
                generated_text=self.tokenizer.decode(
                    all_seqs[i].generated_ids, skip_special_tokens=True,
                ),
                token_ids=all_seqs[i].generated_ids,
            )
            for i in range(len(prompts))
        ]
