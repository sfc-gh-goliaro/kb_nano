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
from transformers import AutoTokenizer

from .context import (
    AttnBackendConfig, get_attn_backend_config, get_context,
    reset_context, set_context, set_mixed_context,
)
from ..tasks.baseline.L1.allreduce import set_custom_ar
from .weight_loader import load_model
from .tp import init_parallel_groups

MAX_MODEL_LEN = 131072
NCCL_PORT = int(os.environ.get("KB_NANO_NCCL_PORT", "29501"))



def _detect_scheduling_defaults() -> tuple[int, int]:
    """Choose max_num_batched_tokens and max_num_seqs based on GPU memory.

    Mirrors vLLM's heuristic: high-memory GPUs (>=70 GiB, non-A100) get
    larger defaults; everything else gets conservative values.
    """
    if not torch.cuda.is_available():
        return 8192, 256
    _GiB = 1 << 30
    _, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0).lower()
    if total >= 70 * _GiB and "a100" not in name:
        return 16384, 1024
    return 8192, 256


_DEFAULT_MAX_NUM_BATCHED_TOKENS, _DEFAULT_MAX_NUM_SEQS = (
    _detect_scheduling_defaults()
)

_PROFILE = os.environ.get("KB_NANO_PROFILE", "0") == "1"


ATTN_BACKEND_CONFIG = get_attn_backend_config()
USE_TRTLLM = ATTN_BACKEND_CONFIG.use_trtllm
BLOCK_SIZE = ATTN_BACKEND_CONFIG.block_size
USE_FLASHINFER = USE_TRTLLM  # back-compat alias


def _reconfigure_attn_backend(cfg: AttnBackendConfig) -> None:
    """Update module-level globals after detecting model-specific backend."""
    global ATTN_BACKEND_CONFIG, USE_TRTLLM, BLOCK_SIZE, USE_FLASHINFER
    from .context import set_attn_backend_config
    set_attn_backend_config(cfg)
    ATTN_BACKEND_CONFIG = cfg
    USE_TRTLLM = cfg.use_trtllm
    BLOCK_SIZE = cfg.block_size
    USE_FLASHINFER = USE_TRTLLM


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

    def preempt(self):
        """Reset to re-prefillable state (vLLM-style recompute preemption)."""
        self.token_ids = list(self.prompt_ids)
        self.generated_ids.clear()
        self.block_table.clear()
        self.num_computed_tokens = 0
        self.status = SeqStatus.WAITING

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.generated_ids.append(token_id)

    def __getstate__(self):
        """Minimal pickling for shared memory transfer to non-rank-0 workers."""
        return (len(self), len(self.prompt_ids), self.block_table,
                self.num_computed_tokens,
                self.token_ids if not self.generated_ids else self.last_token)

    def __setstate__(self, state):
        self._num_tokens, num_prompt, self.block_table, self.num_computed_tokens = state[:-1]
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
        self._num_blocks = num_blocks
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def reset(self):
        """Return all blocks to the free pool."""
        self.free_block_ids = deque(range(self._num_blocks))

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
                 event, shm_name: str,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int = MAX_MODEL_LEN,
                 max_num_seqs: int | None = None,
                 max_num_batched_tokens: int | None = None,
                 data_parallel_size: int = 1,
                 enable_expert_parallel: bool = False):
        self.model_name = model_name
        self.rank = rank
        self.world_size = world_size
        self.enforce_eager = enforce_eager or enable_expert_parallel
        self.event = event
        self.block_size = BLOCK_SIZE
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = ((max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 2) * BLOCK_SIZE
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else _DEFAULT_MAX_NUM_SEQS
        self.max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else _DEFAULT_MAX_NUM_BATCHED_TOKENS
        self.data_parallel_size = data_parallel_size
        self.enable_expert_parallel = enable_expert_parallel

        tp_size = world_size // data_parallel_size

        torch.cuda.set_device(rank)
        self._dist_pre_initialized = dist.is_initialized()
        if not self._dist_pre_initialized:
            dist.init_process_group(
                "nccl", f"tcp://localhost:{NCCL_PORT}",
                world_size=world_size, rank=rank,
                device_id=torch.device(f"cuda:{rank}"),
            )
            init_parallel_groups(
                tp_size=tp_size,
                dp_size=data_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
            )

        self.custom_ar = None
        if tp_size > 1:
            from .tp import get_tp_group
            tp_group = get_tp_group()
            self.cpu_group = dist.new_group(
                ranks=list(range(
                    (rank // tp_size) * tp_size,
                    (rank // tp_size) * tp_size + tp_size,
                )),
                backend="gloo",
            )
            if not os.environ.get("KB_NANO_DISABLE_CUSTOM_AR", "0") == "1":
                from ..tasks.baseline.L1.allreduce import CustomAllreduce
                self.custom_ar = CustomAllreduce(
                    self.cpu_group, rank % tp_size, max_size=8 * 1024 * 1024
                )
                set_custom_ar(self.custom_ar)

        self.dtype = dtype
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")

        from .weight_loader import _detect_model_type
        _early_model_type = _detect_model_type(model_name)
        _early_is_v32 = _early_model_type == "deepseek_v32"
        if _early_is_v32:
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 9:
                _reconfigure_attn_backend(AttnBackendConfig.flashmla_sparse())
                self.block_size = BLOCK_SIZE
                self.max_model_len = ((max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 2) * BLOCK_SIZE
                if self.rank == 0:
                    print(f"  DeepSeek V3.2: FlashMLA sparse backend (block_size={BLOCK_SIZE})")

        self.model, self.config = load_model(
            model_name, torch.device(f"cuda:{rank}"), dtype,
        )
        self.is_moe = hasattr(self.config, "num_local_experts") or hasattr(self.config, "n_routed_experts")
        self.is_deepseek = hasattr(self.config, "qk_nope_head_dim")
        self.is_deepseek_v32 = (self.is_deepseek
                                and hasattr(self.config, "index_topk")
                                and self.config.index_topk is not None)
        self.is_qwen_vl = hasattr(self.config, "mrope_section")
        torch.cuda.synchronize()
        if self.rank == 0:
            _GiB = 1 << 30
            mem = torch.cuda.memory_allocated() / _GiB
            peak = torch.cuda.max_memory_allocated() / _GiB
            print(f"  After model load: allocated={mem:.1f}G peak={peak:.1f}G")
        self._share_trtllm_workspace()
        self._share_activation_buffers()
        if self.rank == 0:
            _GiB = 1 << 30
            print(f"  After sharing bufs: allocated={torch.cuda.memory_allocated()/_GiB:.1f}G")
        self._warmup_deepgemm()
        torch.cuda.synchronize()
        if self.rank == 0:
            _GiB = 1 << 30
            print(f"  After DG warmup: allocated={torch.cuda.memory_allocated()/_GiB:.1f}G")
        self.warmup_model()
        if self.rank == 0:
            _GiB = 1 << 30
            print(f"  After model warmup: allocated={torch.cuda.memory_allocated()/_GiB:.1f}G")
        self._presize_moe_buffers()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        self._init_greedy_buffers()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # TP shared memory setup (only needed when tp_size > 1)
        if tp_size > 1:
            tp_rank = rank % tp_size
            if tp_rank == 0:
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
        tp_size = self.world_size // self.data_parallel_size
        if tp_size > 1:
            self.shm.close()
            dist.barrier()
            tp_rank = self.rank % tp_size
            if tp_rank == 0:
                self.shm.unlink()
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if not self._dist_pre_initialized:
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

    def _share_trtllm_workspace(self):
        """Replace per-layer TRTLLM workspace buffers with a single shared one."""
        if not ATTN_BACKEND_CONFIG.use_trtllm:
            return
        self._attn_layers = []
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                self._attn_layers.append(module)
        trtllm_workspace = torch.zeros(
            512 * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{self.rank}"
        )
        for layer in self._attn_layers:
            layer.set_trtllm_workspace(trtllm_workspace)
        torch.cuda.empty_cache()

    def _share_activation_buffers(self):
        """Share activation buffers across all layers that execute sequentially.

        Covers SiluAndMul, FusedExperts intermediate caches, and FP8 quantization
        buffers. Must be called before warmup so only one buffer grows to max size
        instead of one per layer.
        """
        from ..tasks.baseline.L1.silu_and_mul import SiluAndMul
        silu_modules = [
            m for m in self.model.modules() if isinstance(m, SiluAndMul)
        ]
        if len(silu_modules) > 1:
            shared = silu_modules[0]._act_buf
            for m in silu_modules[1:]:
                m.set_shared_buffer(shared)

        from ..tasks.baseline.L2.fused_experts import FusedExperts
        fused_experts = [
            m for m in self.model.modules() if isinstance(m, FusedExperts)
        ]
        if len(fused_experts) > 1:
            shared_bufs = fused_experts[0]._shared_bufs
            for m in fused_experts[1:]:
                m.set_shared_bufs(shared_bufs)

        from ..tasks.baseline.L1.moe_sum import MoeSum
        moe_sums = [m for m in self.model.modules() if isinstance(m, MoeSum)]
        if len(moe_sums) > 1:
            shared_buf = moe_sums[0]._buf
            for m in moe_sums[1:]:
                m.set_shared_buf(shared_buf)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if self.is_qwen_vl:
            self._warmup_vision_encoder()

        warmup_len = min(self.max_model_len, self.max_num_batched_tokens)
        num_seqs = min(self.max_num_batched_tokens // warmup_len, self.max_num_seqs)

        from .tp import _ep_size, get_ep_group
        ep = _ep_size()
        if ep > 1:
            warmup_len = min(warmup_len, max(16, self.max_num_batched_tokens // (ep * ep)))
            num_seqs = 1
            ep_group = get_ep_group()
            if ep_group is not None:
                dist.barrier(ep_group)

        if self.rank == 0:
            print(f"  Warmup: {num_seqs} seq(s) of length {warmup_len}, ep={ep}")
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        self.run(seqs, True)

        torch.cuda.empty_cache()

    def _presize_moe_buffers(self):
        """Pre-allocate MoE intermediate and FP8 buffers to their maximum
        inference-time size.

        With FlashInfer, intermediate buffers are managed internally so we
        only pre-allocate EP gather/scatter buffers. With the Triton fallback,
        we pre-allocate shared intermediate buffers for the grouped GEMMs.
        """
        from ..tasks.baseline.L2.fused_experts import FusedExperts, use_flashinfer_cutlass
        from .tp import _ep_size

        fused = [m for m in self.model.modules() if isinstance(m, FusedExperts)]
        if not fused:
            return

        ep = _ep_size()
        use_fi = use_flashinfer_cutlass()
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        cfg = self.config
        K = cfg.hidden_size
        N2 = 2 * getattr(cfg, 'moe_intermediate_size', getattr(cfg, 'intermediate_size', K))
        top_k = getattr(cfg, 'num_experts_per_tok', 8)

        if not use_fi:
            if ep > 1:
                M = 256 * ep
            else:
                M = self.max_num_batched_tokens

            bufs = fused[0]._shared_bufs
            bufs.get_cache("cache1", (M * top_k, N2), device, dtype)
            bufs.get_cache("cache3", (M * top_k, K), device, dtype)

            is_fp8 = any(
                hasattr(m, 'w13') and m.w13 is not None and m.w13.dtype == torch.float8_e4m3fn
                for m in self.model.modules()
                if hasattr(m, 'w13')
            )
            if is_fp8:
                fused[0]._ensure_fp8_bufs(M, K, N2, top_k, device)

            from ..tasks.baseline.L1.silu_and_mul import SiluAndMul
            silu_modules = [m for m in self.model.modules() if isinstance(m, SiluAndMul)]
            if silu_modules:
                half = N2 // 2
                silu_modules[0]._act_buf.get(M * top_k, half, device, dtype)

            from ..tasks.baseline.L1.moe_sum import MoeSum
            moe_sums = [m for m in self.model.modules() if isinstance(m, MoeSum)]
            if moe_sums:
                moe_sums[0]._buf.get(M, K, device, dtype)

        if ep > 1 and self.enable_expert_parallel:
            self._presize_ep_buffers(ep, K, top_k, device, dtype)

        torch.cuda.synchronize()
        if self.rank == 0:
            _GiB = 1 << 30
            mem = torch.cuda.memory_allocated() / _GiB
            print(f"  After MoE buffer pre-sizing (top_k={top_k}, ep={ep}, "
                  f"flashinfer={use_fi}): allocated={mem:.1f}G")

    def _presize_ep_buffers(self, ep_size, D, top_k, device, dtype):
        """Pre-allocate shared EP gather/scatter buffers for the decode path.

        All MoE layers share one set of buffers (they execute sequentially).
        Sized for max_num_seqs per EP rank.
        """
        from ..tasks.baseline.L2.deepseek_moe import DeepSeekMoE
        max_n = self.max_num_seqs
        total_gathered = ep_size * max_n

        buf_h = torch.empty(total_gathered, D, dtype=dtype, device=device)
        buf_w = torch.empty(total_gathered, top_k, dtype=dtype, device=device)
        buf_i = torch.empty(total_gathered, top_k, dtype=torch.int64, device=device)
        chunk_h = torch.empty(max_n, D, dtype=dtype, device=device)
        chunk_w = torch.empty(max_n, top_k, dtype=dtype, device=device)
        chunk_i = torch.empty(max_n, top_k, dtype=torch.int64, device=device)
        rs_out = torch.empty(max_n, D, dtype=dtype, device=device)
        gather_h = [buf_h[i * max_n:(i + 1) * max_n] for i in range(ep_size)]
        gather_w = [buf_w[i * max_n:(i + 1) * max_n] for i in range(ep_size)]
        gather_i = [buf_i[i * max_n:(i + 1) * max_n] for i in range(ep_size)]

        for m in self.model.modules():
            if not isinstance(m, DeepSeekMoE):
                continue
            m._ep_buf_h = buf_h
            m._ep_buf_w = buf_w
            m._ep_buf_i = buf_i
            m._ep_chunk_h = chunk_h
            m._ep_chunk_w = chunk_w
            m._ep_chunk_i = chunk_i
            m._ep_rs_out = rs_out
            m._ep_gather_h = gather_h
            m._ep_gather_w = gather_w
            m._ep_gather_i = gather_i
            m._ep_bufs_initialized = True

    def _warmup_deepgemm(self):
        """Pre-JIT DeepGEMM FP8 kernels for all weight shapes at decode and
        prefill batch sizes, pre-allocate decode buffers per instance and
        shared prefill buffers per unique (K, N) shape."""
        from ..tasks.baseline.L1.fp8_linear import Fp8Linear, _Fp8PrefillBufs
        import deep_gemm
        torch.cuda.synchronize()
        if self.rank == 0:
            print("  Starting DeepGEMM warmup...")

        fp8_modules = []
        for module in self.model.modules():
            linear_op = getattr(module, 'linear_op', None)
            if isinstance(linear_op, Fp8Linear):
                fp8_modules.append((module, linear_op))

        if not fp8_modules:
            return

        max_decode = self.max_num_seqs
        max_prefill = self.max_num_batched_tokens
        device = next(self.model.parameters()).device

        decode_bs = [1, 2, 4, 8] + list(range(16, max_decode + 1, 16))
        prefill_bs = [s for s in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
                      if s <= max_prefill and s > max_decode]

        prefill_bufs: dict[tuple[int, int], _Fp8PrefillBufs] = {}

        seen_shapes = set()
        for module, linear_op in fp8_modules:
            w = module.weight
            ws = module.weight_scale_inv
            N, K = w.shape

            linear_op._ensure_buffers(max_decode, K, N, device)
            linear_op._a_buf.zero_()
            linear_op._s_buf.fill_(1.0)
            linear_op._o_buf.zero_()

            key = (N, K)
            if key not in prefill_bufs:
                pf_buf = _Fp8PrefillBufs(max_prefill, K, N, device)
                pf_buf.a.zero_()
                pf_buf.s.fill_(1.0)
                pf_buf.o.zero_()
                prefill_bufs[key] = pf_buf
            linear_op._pf = prefill_bufs[key]

            if key in seen_shapes:
                continue
            seen_shapes.add(key)

            a_fp8 = linear_op._a_buf
            a_scale = linear_op._s_buf
            out = linear_op._o_buf
            for num_tokens in decode_bs:
                if num_tokens > max_decode:
                    break
                deep_gemm.fp8_gemm_nt(
                    (a_fp8[:num_tokens], a_scale[:num_tokens]),
                    (w, ws),
                    out[:num_tokens],
                )

            pf = prefill_bufs[key]
            for num_tokens in prefill_bs:
                deep_gemm.fp8_gemm_nt(
                    (pf.a[:num_tokens], pf.s[:num_tokens]),
                    (w, ws),
                    pf.o[:num_tokens],
                )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if self.rank == 0:
            print(f"  DeepGEMM warmup: {len(seen_shapes)} unique FP8 weight shapes")

    def _warmup_vision_encoder(self):
        """Run vision encoder with worst-case dummy inputs to capture peak
        activation memory, following vLLM's profile_run() approach."""
        import math
        from PIL import Image

        model = self.model
        vision_cfg = self.config.vision
        patch_size = vision_cfg.patch_size
        merge_size = vision_cfg.spatial_merge_size
        temporal_patch_size = getattr(vision_cfg, "temporal_patch_size", 2)

        max_pixels = getattr(vision_cfg, "max_pixels", None)
        if max_pixels is None:
            max_pixels = 1280 * 28 * 28

        unit = patch_size * merge_size
        max_patches = max_pixels // (unit * unit)

        def _closest_factor_pair(n):
            for d in range(math.isqrt(n), 0, -1):
                if n % d == 0:
                    return d, n // d
            return 1, n

        hf, wf = 1, max_patches
        for s in range(max_patches, 0, -1):
            hf, wf = _closest_factor_pair(s)
            if wf / hf <= 200:
                break
        img_h, img_w = unit * hf, unit * wf

        if self.rank == 0:
            print(f"  Vision warmup: image {img_w}x{img_h}")

        dummy_img = Image.new("RGB", (img_w, img_h), color=255)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": dummy_img},
            {"type": "text", "text": "x"},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[dummy_img], return_tensors="pt", padding=True)

        pv = inputs["pixel_values"].cuda()
        grid_thw = inputs["image_grid_thw"].cpu()
        with torch.inference_mode():
            vis_out = model.visual(pv, grid_thw=grid_thw)

        self._warmup_encoder_cache = {}
        if isinstance(vis_out, tuple):
            embeds, ds = vis_out
            self._warmup_encoder_cache["img"] = (embeds, ds)
        else:
            self._warmup_encoder_cache["img"] = vis_out

        num_frames = 32
        padded_nf = num_frames + num_frames % temporal_patch_size
        vid_h = min(img_h, 420)
        vid_w = min(img_w, 420)
        vid_h = (vid_h // unit) * unit or unit
        vid_w = (vid_w // unit) * unit or unit

        if self.rank == 0:
            print(f"  Vision warmup: video {num_frames}f {vid_w}x{vid_h}")

        dummy_vid = np.full(
            (num_frames, vid_h, vid_w, 3), 255, dtype=np.uint8)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": list(dummy_vid)},
            {"type": "text", "text": "x"},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        vid_frames_pil = [Image.fromarray(f) for f in dummy_vid]
        inputs = processor(
            text=[text], videos=[vid_frames_pil],
            return_tensors="pt", padding=True)

        vpv = inputs["pixel_values_videos"].cuda()
        vgrid = inputs["video_grid_thw"].cpu()
        with torch.inference_mode():
            vis_out = model.visual(vpv, grid_thw=vgrid)

        if isinstance(vis_out, tuple):
            embeds, ds = vis_out
            self._warmup_encoder_cache["vid"] = (embeds, ds)
        else:
            self._warmup_encoder_cache["vid"] = vis_out

        del processor, dummy_img, dummy_vid, vid_frames_pil, pv, vpv
        del inputs, messages, text

    def allocate_kv_cache(self):
        if not hasattr(self, '_attn_layers') or not self._attn_layers:
            self._attn_layers = []
            self._indexer_layers = []
            from ..tasks.baseline.L2.deepseek_indexer import DeepSeekIndexer
            for module in self.model.modules():
                if isinstance(module, DeepSeekIndexer):
                    self._indexer_layers.append(module)
                elif hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    self._attn_layers.append(module)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        from .tp import _tp_size
        tp = _tp_size()

        num_layers = self.config.num_hidden_layers

        if self.is_deepseek_v32 and ATTN_BACKEND_CONFIG.use_flashmla_sparse:
            mla_bytes_per_tok = 656
            indexer_head_dim = getattr(self.config, 'index_head_dim', 128)
            indexer_bytes_per_tok = indexer_head_dim + 4

            block_bytes = num_layers * BLOCK_SIZE * (mla_bytes_per_tok + indexer_bytes_per_tok)

            non_torch = used - current
            utilization = min(self.gpu_memory_utilization, 0.90)
            available = total * utilization - non_torch - current

            if self.rank == 0:
                _GiB = 1 << 30
                print(f"  Memory: total={total/_GiB:.1f}G free={free/_GiB:.1f}G "
                      f"used={used/_GiB:.1f}G peak={peak/_GiB:.1f}G "
                      f"current={current/_GiB:.1f}G non_torch={non_torch/_GiB:.1f}G "
                      f"avail={available/_GiB:.1f}G block_bytes={block_bytes}")

            num_blocks = int(available) // block_bytes
            assert num_blocks > 0, f"Not enough GPU memory for KV cache on rank {self.rank}"
            self.num_blocks = num_blocks

            if self.rank == 0:
                print(f"  KV cache: {num_blocks} blocks x {BLOCK_SIZE} = "
                      f"{num_blocks * BLOCK_SIZE} token slots (FP8 MLA + indexer)")

            mla_cache = torch.zeros(
                num_layers, num_blocks, BLOCK_SIZE, mla_bytes_per_tok,
                dtype=torch.uint8, device="cuda",
            )
            indexer_cache = torch.zeros(
                num_layers, num_blocks, BLOCK_SIZE, indexer_bytes_per_tok,
                dtype=torch.uint8, device="cuda",
            )

            layer_id = 0
            for module in self._attn_layers:
                module.k_cache = mla_cache[layer_id]
                module.v_cache = mla_cache[layer_id]
                layer_id += 1

            idx_layer_id = 0
            for module in self._indexer_layers:
                module.k_cache = indexer_cache[idx_layer_id]
                idx_layer_id += 1

            self.kv_cache = mla_cache
            self._indexer_cache = indexer_cache

        else:
            if self.is_deepseek:
                num_kv_heads = 1
                head_dim = self.config.kv_lora_rank + self.config.qk_rope_head_dim
            else:
                num_kv_heads = self.config.num_key_value_heads // tp
                head_dim = self.config.head_dim

            elem_size = torch.finfo(torch.get_default_dtype()).bits // 8
            block_bytes = 2 * num_layers * BLOCK_SIZE * num_kv_heads * head_dim * elem_size
            non_torch = used - current
            utilization = self.gpu_memory_utilization
            if self.is_deepseek:
                utilization = min(utilization, 0.88)
            available = total * utilization - non_torch - current
            if self.rank == 0:
                _GiB = 1 << 30
                print(f"  Memory: total={total/_GiB:.1f}G free={free/_GiB:.1f}G "
                      f"used={used/_GiB:.1f}G peak={peak/_GiB:.1f}G "
                      f"current={current/_GiB:.1f}G non_torch={non_torch/_GiB:.1f}G "
                      f"avail={available/_GiB:.1f}G block_bytes={block_bytes}")
            num_blocks = int(available) // block_bytes
            if self.is_qwen_vl:
                num_blocks = int(num_blocks * 0.95)
            assert num_blocks > 0, f"Not enough GPU memory for KV cache on rank {self.rank}"
            self.num_blocks = num_blocks
            if self.rank == 0:
                print(f"  KV cache: {num_blocks} blocks x {BLOCK_SIZE} = {num_blocks * BLOCK_SIZE} token slots")

            if ATTN_BACKEND_CONFIG.kv_layout == "HND":
                self.kv_cache = torch.empty(
                    2, num_layers, num_blocks, num_kv_heads, BLOCK_SIZE, head_dim,
                )
            else:
                self.kv_cache = torch.empty(
                    2, num_layers, num_blocks, BLOCK_SIZE, num_kv_heads, head_dim,
                )
            layer_id = 0
            for module in self._attn_layers:
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        if self.rank == 0:
            cfg = ATTN_BACKEND_CONFIG
            print(f"  Attention backend: {cfg.backend} "
                  f"(block_size={cfg.block_size}, kv_layout={cfg.kv_layout})")

        if hasattr(self, '_warmup_encoder_cache'):
            del self._warmup_encoder_cache
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def prepare_prefill(self, seqs):
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_sq, max_sk = 0, 0
        slot_mapping = []
        max_bt = 0
        has_block_tables = False
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
            has_block_tables = True
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
        max_cl = int(cl[:n].max())
        set_context(
            False,
            slot_mapping=torch.from_numpy(sm).pin_memory().cuda(non_blocking=True),
            context_lens=torch.from_numpy(cl).pin_memory().cuda(non_blocking=True),
            block_tables=torch.from_numpy(bt).pin_memory().cuda(non_blocking=True),
            max_context_len=max_cl,
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
        """Prepare a unified mixed batch: [prefill_tokens... | decode_tokens...].

        The attention layer receives split metadata so it can dispatch
        prefill tokens to the prefill kernel and decode tokens to the
        decode kernel independently.
        """
        input_ids, positions = [], []
        slot_mapping = []
        block_size = self.block_size

        use_mrope = self.is_qwen_vl and any(
            getattr(s, 'mrope_positions', None) is not None for s in prefill_seqs)
        mrope_pos_list = [] if use_mrope else None

        # --- Prefill portion ---
        pf_cu_q, pf_cu_k = [0], [0]
        pf_max_sq, pf_max_sk = 0, 0
        pf_max_bt = 0

        for seq, chunk_size in zip(prefill_seqs, prefill_chunk_sizes):
            start_pos = seq.num_computed_tokens
            chunk_ids = seq.token_ids[start_pos:start_pos + chunk_size]
            input_ids.extend(chunk_ids)
            if use_mrope and seq.mrope_positions is not None:
                mrope_pos_list.append(seq.mrope_positions[:, start_pos:start_pos + chunk_size])
            else:
                positions.extend(range(start_pos, start_pos + chunk_size))
                if use_mrope:
                    mrope_pos_list.append(
                        torch.arange(start_pos, start_pos + chunk_size,
                                     dtype=torch.int64).unsqueeze(0).expand(3, -1)
                    )

            kv_len = start_pos + chunk_size
            pf_cu_q.append(pf_cu_q[-1] + chunk_size)
            pf_cu_k.append(pf_cu_k[-1] + kv_len)
            pf_max_sq = max(chunk_size, pf_max_sq)
            pf_max_sk = max(kv_len, pf_max_sk)

            for p in range(start_pos, start_pos + chunk_size):
                slot_mapping.append(
                    seq.block_table[p // block_size] * block_size + (p % block_size)
                )
            blen = len(seq.block_table)
            if blen > pf_max_bt:
                pf_max_bt = blen

        num_prefill_tokens = len(input_ids)
        num_prefill_seqs = len(prefill_seqs)

        # Build prefill block table
        prefill_block_tables = None
        if pf_max_bt > 0 and num_prefill_seqs > 0:
            pbt = np.full((num_prefill_seqs, pf_max_bt), -1, dtype=np.int32)
            for i, seq in enumerate(prefill_seqs):
                b = seq.block_table
                pbt[i, :len(b)] = b
            prefill_block_tables = torch.from_numpy(pbt).pin_memory().cuda(non_blocking=True)

        # --- Decode portion ---
        nd = len(decode_seqs)
        dc_cl = np.empty(nd, dtype=np.int32)
        dc_max_bt = 0
        for i, seq in enumerate(decode_seqs):
            input_ids.append(seq.last_token)
            pos = len(seq) - 1
            if use_mrope:
                delta = getattr(seq, 'mrope_position_delta', 0)
                p = pos + delta
                mrope_pos_list.append(
                    torch.tensor([[p], [p], [p]], dtype=torch.int64))
            else:
                positions.append(pos)
            dc_cl[i] = len(seq)
            slot_mapping.append(
                seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
            )
            blen = len(seq.block_table)
            if blen > dc_max_bt:
                dc_max_bt = blen

        dc_bt = np.full((nd, dc_max_bt), -1, dtype=np.int32) if nd > 0 else np.empty((0, 0), dtype=np.int32)
        for i, seq in enumerate(decode_seqs):
            b = seq.block_table
            dc_bt[i, :len(b)] = b
        dc_max_cl = int(dc_cl[:nd].max()) if nd > 0 else 0

        logit_idx = []
        for i in range(num_prefill_seqs):
            logit_idx.append(pf_cu_q[i + 1] - 1)
        for j in range(nd):
            logit_idx.append(num_prefill_tokens + j)

        set_mixed_context(
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=nd,
            num_prefill_seqs=num_prefill_seqs,
            prefill_cu_seqlens_q=torch.tensor(pf_cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            prefill_cu_seqlens_k=torch.tensor(pf_cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            prefill_max_seqlen_q=pf_max_sq,
            prefill_max_seqlen_k=pf_max_sk,
            prefill_block_tables=prefill_block_tables,
            decode_context_lens=torch.from_numpy(dc_cl).pin_memory().cuda(non_blocking=True) if nd > 0 else None,
            decode_block_tables=torch.from_numpy(dc_bt).pin_memory().cuda(non_blocking=True) if nd > 0 else None,
            decode_max_context_len=dc_max_cl,
            logit_indices=torch.tensor(logit_idx, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
        )

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if use_mrope:
            positions_t = torch.cat(mrope_pos_list, dim=1).to(torch.int64).pin_memory().cuda(non_blocking=True)
        else:
            positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        return input_ids_t, positions_t

    @torch.inference_mode()
    def run_model(self, input_ids, positions, is_prefill, inputs_embeds=None,
                  deepstack_embeds=None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs_list[-1]:
            if self.enable_expert_parallel:
                from ..tasks.baseline.L2.deepseek_moe import set_ep_max_n
                set_ep_max_n(None)
            if inputs_embeds is not None:
                return self.model.compute_logits(
                    self.model(input_ids, positions, inputs_embeds=inputs_embeds,
                               deepstack_embeds=deepstack_embeds)
                )
            return self.model.compute_logits(self.model(input_ids, positions))
        bs = input_ids.size(0)
        ctx = get_context()
        graph_bs = self._graph_bs_for_n[bs]
        if self.enable_expert_parallel:
            from ..tasks.baseline.L2.deepseek_moe import set_ep_max_n
            set_ep_max_n(graph_bs)
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

        self._run_graph_from_numpy(n, ids_np, pos_np, sm_np, cl_np, bt_np)
        return self._greedy_from_hidden(n)

    def _run_graph_from_numpy(self, n, ids_np, pos_np, sm_np, cl_np, bt_np):
        """Copy numpy arrays into graph vars and replay the CUDA graph."""
        gv = self.graph_vars
        graph_bs = self._graph_bs_for_n[n]
        prev_n = getattr(self, '_prev_decode_n', -1)

        if self.enable_expert_parallel:
            from ..tasks.baseline.L2.deepseek_moe import set_ep_max_n
            set_ep_max_n(graph_bs)

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

    @torch.inference_mode()
    def run_decode_greedy_fast_async(self, decode_data):
        """Like run_decode_greedy_fast but starts async D2H copy.

        Returns (has_result, n) -- caller must call _wait_async_tokens(n)
        later to get the Python list of token IDs.
        """
        n, ids_np, pos_np, sm_np, cl_np, bt_np = decode_data

        if self.enforce_eager:
            result = self._run_decode_greedy_eager(n, ids_np, pos_np, sm_np, cl_np, bt_np)
            if result is not None:
                main_stream = torch.cuda.current_stream()
                cs = self._copy_stream
                with torch.cuda.stream(cs):
                    cs.wait_stream(main_stream)
                    self._pinned_token_ids[:n].copy_(result, non_blocking=True)
                    self._copy_event.record(cs)
                return True, n
            return False, n

        self._run_graph_from_numpy(n, ids_np, pos_np, sm_np, cl_np, bt_np)
        has_result = self._greedy_from_hidden_async(n)
        return has_result, n

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
            max_context_len=int(cl_np.max()),
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
        max_bs = self.max_num_seqs
        dev = f"cuda:{self.rank}"
        self._greedy_info = torch.zeros(max_bs, 2, dtype=torch.float32, device=dev)
        self._greedy_gathered = [
            torch.zeros(max_bs, 2, dtype=torch.float32, device=dev)
            for _ in range(self.world_size)
        ]
        self._greedy_all_info = torch.zeros(self.world_size, max_bs, 2, dtype=torch.float32, device=dev)
        self._greedy_arange = torch.arange(max_bs, device=dev)

        max_num_blocks = (self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE
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

        # Async D2H: pinned buffer + copy stream for pipelined decode
        self._pinned_token_ids = torch.empty(max_bs, dtype=torch.int64,
                                             device="cpu", pin_memory=True)
        self._copy_stream = torch.cuda.Stream(device=dev)
        self._copy_event = torch.cuda.Event()

    def _greedy_from_hidden(self, n):
        """Use CUDA-graph-captured LM head + local argmax, then allgather.
        
        Returns GPU tensor of token IDs (rank 0) or None (other ranks).
        Caller must call .tolist() to sync.
        """
        gv = self.graph_vars

        if self.world_size == 1:
            return gv["lm_max_idxs"][:n]

        local_max_vals = gv["lm_max_vals"][:n]
        local_max_idxs = gv["lm_max_idxs"][:n] + self.model.lm_head.vocab_start

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

    def _greedy_from_hidden_async(self, n):
        """Like _greedy_from_hidden but starts async D2H copy.

        After calling this, the caller must eventually call
        _wait_async_tokens(n) to get the Python list of token IDs.
        Between the two calls, the CPU is free to do other work.
        """
        gpu_ids = self._greedy_from_hidden(n)
        if gpu_ids is not None:
            main_stream = torch.cuda.current_stream()
            cs = self._copy_stream
            with torch.cuda.stream(cs):
                cs.wait_stream(main_stream)
                self._pinned_token_ids[:n].copy_(gpu_ids, non_blocking=True)
                self._copy_event.record(cs)
        return gpu_ids is not None

    def _wait_async_tokens(self, n):
        """Wait for the async D2H copy to complete and return token list."""
        self._copy_event.synchronize()
        return self._pinned_token_ids[:n].tolist()

    def _prepare_decode_arrays(self, seqs):
        """Precompute numpy arrays for decode - uses pre-allocated buffers."""
        n = len(seqs)
        ids_np = self._np_ids
        pos_np = self._np_pos
        sm_np = self._np_sm
        cl_np = self._np_cl
        max_bt = 0
        bs = BLOCK_SIZE
        for i, seq in enumerate(seqs):
            tids = seq.token_ids
            if tids is not None:
                slen = len(tids)
                ids_np[i] = tids[-1]
            else:
                slen = seq._num_tokens
                ids_np[i] = seq._last_token
            if self.is_qwen_vl:
                pos_np[i] = slen - 1 + seq.mrope_position_delta
            else:
                pos_np[i] = slen - 1
            cl_np[i] = slen
            bt = seq.block_table
            blen = len(bt)
            r = slen % bs
            sm_np[i] = bt[-1] * bs + (r - 1 if r else bs - 1)
            if blen > max_bt:
                max_bt = blen
        bt_np = self._np_bt
        for i, seq in enumerate(seqs):
            b = seq.block_table
            blen = len(b)
            bt_np[i, :blen] = b
            if blen < max_bt:
                bt_np[i, blen:max_bt] = -1
        self._prev_max_bt = max_bt
        return (n, ids_np[:n], pos_np[:n], sm_np[:n], cl_np[:n], bt_np[:n, :max_bt])

    def _update_decode_arrays_incremental(self, n, token_ids, decode_seqs):
        """Update pre-allocated decode arrays incrementally after a decode step.

        Much faster than _prepare_decode_arrays: vectorized numpy ops +
        only touches block table rows that crossed a block boundary.
        """
        ids_np = self._np_ids
        pos_np = self._np_pos
        sm_np = self._np_sm
        cl_np = self._np_cl
        bt_np = self._np_bt
        bs = BLOCK_SIZE

        ids_np[:n] = token_ids
        pos_np[:n] += 1
        cl_np[:n] += 1
        sm_np[:n] += 1

        boundary_mask = cl_np[:n] % bs == 1
        if boundary_mask.any():
            max_bt = 0
            for i in np.where(boundary_mask)[0]:
                seq = decode_seqs[i]
                bt = seq.block_table
                blen = len(bt)
                sm_np[i] = bt[-1] * bs
                bt_np[i, :blen] = bt
                if blen > max_bt:
                    max_bt = blen
            if max_bt == 0:
                max_bt = self._prev_max_bt
            else:
                for i in np.where(~boundary_mask)[0]:
                    blen = len(decode_seqs[i].block_table)
                    if blen > max_bt:
                        max_bt = blen
                self._prev_max_bt = max_bt
        else:
            max_bt = self._prev_max_bt
        return (n, ids_np[:n], pos_np[:n], sm_np[:n], cl_np[:n],
                bt_np[:n, :max_bt])

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

    def call_decode_greedy_async(self, decode_data):
        """Launch greedy decode from precomputed arrays and start async D2H.

        Returns (has_result, n). Caller must call
        model_runner._wait_async_tokens(n) to get token IDs.
        """
        n = decode_data[0]
        if self.world_size > 1 and self.rank == 0:
            self._write_decode_shm(*decode_data)
            self.shm.buf[self._SHM_FLAG_OFFSET] = 1
            self._signal_workers()
            return self.run_decode_greedy_fast_async(decode_data)
        return self.run_decode_greedy_fast_async(decode_data)

    def run(self, seqs, is_prefill):
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
        max_bs = self.max_num_seqs
        max_num_blocks = (self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE
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

        use_ep = self.enable_expert_parallel
        if use_ep:
            from ..tasks.baseline.L2.deepseek_moe import set_ep_max_n

        ar_ctx = self.custom_ar.capture() if self.custom_ar is not None else nullcontext()
        with ar_ctx:
            for bs in reversed(self.graph_bs_list):
                graph = torch.cuda.CUDAGraph()
                set_context(
                    False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                    max_context_len=self.max_model_len,
                )
                if use_ep:
                    set_ep_max_n(bs)
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

        if use_ep:
            set_ep_max_n(None)

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
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        data_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
    ):
        self.model_name = model_name
        self.seed = seed
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else _DEFAULT_MAX_NUM_SEQS
        self.max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else _DEFAULT_MAX_NUM_BATCHED_TOKENS
        self.data_parallel_size = data_parallel_size
        self.enable_expert_parallel = enable_expert_parallel
        self._set_seeds(seed)

        world_size = tensor_parallel_size * data_parallel_size

        # Unique shared memory name to avoid collisions
        shm_name = f"sllama_{uuid.uuid4().hex[:8]}"

        mr_kwargs = dict(
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            data_parallel_size=data_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
        )

        # Launch non-rank-0 workers
        self.workers = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, world_size):
            event = ctx.Event()
            p = ctx.Process(
                target=ModelRunner,
                args=(model_name, i, world_size, dtype,
                      enforce_eager, event, shm_name),
                kwargs=mr_kwargs,
            )
            p.start()
            self.workers.append(p)
            self.events.append(event)

        # Rank 0 model runner (events is a list for rank 0)
        self.model_runner = ModelRunner(
            model_name, 0, world_size, dtype,
            enforce_eager, self.events, shm_name,
            **mr_kwargs,
        )
        self.block_manager = BlockManager(self.model_runner.num_blocks)
        print(f"  Scheduling: max_num_seqs={self.max_num_seqs}, "
              f"max_num_batched_tokens={self.max_num_batched_tokens}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.is_qwen_vl = self.model_runner.is_qwen_vl
        self.processor = None
        if self.is_qwen_vl:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name)

        self.encoder_cache: dict[int, tuple] = {}

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
    def _run_vision_encoder(self, seqs, chunk_sizes=None):
        """Run vision encoder with batching and caching.

        Images are batched into a single model.visual() call (concatenated
        pixel_values, stacked grid_thw). Videos are processed one-by-one to
        avoid OOM. Results are cached by id(seq) for reuse across steps.

        If chunk_sizes is provided, only produces embeddings for the chunk
        range [num_computed_tokens : num_computed_tokens + chunk_size] per seq,
        enabling chunked multimodal prefill.

        Returns (inputs_embeds, deepstack_embeds) where deepstack_embeds is a
        list of tensors for Qwen3-VL DeepStack, or None for Qwen2-VL.
        """
        model = self.model_runner.model
        has_deepstack = hasattr(model.visual, 'deepstack_merger_list')
        merge_size = model.config.vision.spatial_merge_size
        image_token_id = self.model_runner.config.image_token_id
        video_token_id = self.model_runner.config.video_token_id

        # --- Phase 1: Run encoder for uncached sequences ---
        img_seqs = []
        img_pv_list = []
        img_thw_list = []
        for seq in seqs:
            if seq.pixel_values is not None and id(seq) not in self.encoder_cache:
                img_seqs.append(seq)
                img_pv_list.append(seq.pixel_values.cuda())
                thw = seq.image_grid_thw
                if not isinstance(thw, torch.Tensor):
                    thw = torch.tensor(thw, dtype=torch.long)
                img_thw_list.append(thw)

        if img_seqs:
            batched_pv = torch.cat(img_pv_list, dim=0)
            batched_thw = torch.cat(img_thw_list, dim=0).cpu()
            vis_out = model.visual(batched_pv, grid_thw=batched_thw)

            if has_deepstack:
                all_img_embeds, all_ds_features = vis_out
            else:
                all_img_embeds = vis_out
                all_ds_features = None

            sizes = (batched_thw.prod(-1) // (merge_size ** 2)).tolist()
            per_seq_embeds = all_img_embeds.split(sizes)
            if all_ds_features is not None:
                per_seq_ds = [ds.split(sizes) for ds in all_ds_features]
            else:
                per_seq_ds = None

            embed_idx = 0
            for i, seq in enumerate(img_seqs):
                thw = seq.image_grid_thw
                n_items = len(thw) if isinstance(thw, list) else thw.shape[0]
                seq_embeds = torch.cat(
                    per_seq_embeds[embed_idx:embed_idx + n_items], dim=0
                ) if n_items > 1 else per_seq_embeds[embed_idx]
                if per_seq_ds is not None:
                    seq_ds = [
                        torch.cat(ds[embed_idx:embed_idx + n_items], dim=0)
                        if n_items > 1 else ds[embed_idx]
                        for ds in per_seq_ds
                    ]
                else:
                    seq_ds = []
                self.encoder_cache[id(seq)] = (seq_embeds, seq_ds, "image")
                embed_idx += n_items

            del batched_pv, batched_thw

        for seq in seqs:
            if seq.video_pixel_values is not None and id(seq) not in self.encoder_cache:
                video_pv = seq.video_pixel_values.cuda()
                grid_thw = seq.video_grid_thw
                if not isinstance(grid_thw, torch.Tensor):
                    grid_thw = torch.tensor(grid_thw, dtype=torch.long)
                grid_thw = grid_thw.cpu()
                vis_out = model.visual(video_pv, grid_thw=grid_thw)

                if has_deepstack:
                    video_embeds, ds_features = vis_out
                else:
                    video_embeds = vis_out
                    ds_features = []
                self.encoder_cache[id(seq)] = (video_embeds, ds_features, "video")
                del video_pv

        # --- Phase 2: Merge vision embeddings into text embeddings ---
        all_inputs_embeds = []
        all_deepstack = [] if has_deepstack else None
        embed_fn = model.get_input_embeddings()

        for seq_idx, seq in enumerate(seqs):
            full_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cuda")

            if chunk_sizes is not None:
                start = seq.num_computed_tokens
                end = start + chunk_sizes[seq_idx]
                chunk_ids = full_ids[start:end]
            else:
                chunk_ids = full_ids
                start = 0
                end = len(full_ids)

            text_embeds = embed_fn(chunk_ids)
            seq_deepstack = [] if has_deepstack else None

            cached = self.encoder_cache.get(id(seq))
            if cached is not None:
                embeds, ds_features, modality = cached
                tok_id = image_token_id if modality == "image" else video_token_id

                mask = chunk_ids == tok_id
                if mask.any():
                    if chunk_sizes is not None:
                        full_mask = full_ids == tok_id
                        chunk_vis_start = full_mask[:start].sum().item()
                        n_vis_in_chunk = mask.sum().item()
                        chunk_embeds = embeds[chunk_vis_start:chunk_vis_start + n_vis_in_chunk]
                        text_embeds[mask] = chunk_embeds.to(text_embeds.dtype)
                    else:
                        text_embeds[mask] = embeds.to(text_embeds.dtype)

                if has_deepstack and ds_features:
                    for i_ds, ds_feat in enumerate(ds_features):
                        ds_expanded = torch.zeros_like(text_embeds)
                        if mask.any():
                            if chunk_sizes is not None:
                                ds_expanded[mask] = ds_feat[chunk_vis_start:chunk_vis_start + n_vis_in_chunk].to(text_embeds.dtype)
                            else:
                                ds_expanded[mask] = ds_feat.to(text_embeds.dtype)
                        if modality == "video" and i_ds < len(seq_deepstack):
                            seq_deepstack[i_ds] = seq_deepstack[i_ds] + ds_expanded
                        else:
                            seq_deepstack.append(ds_expanded)

            all_inputs_embeds.append(text_embeds)
            if has_deepstack:
                all_deepstack.append(seq_deepstack)

        inputs_embeds = torch.cat(all_inputs_embeds, dim=0)

        if has_deepstack and all_deepstack:
            num_levels = max((len(ds) for ds in all_deepstack), default=0)
            if num_levels > 0:
                deepstack_embeds = []
                for level in range(num_levels):
                    level_parts = []
                    for ds in all_deepstack:
                        if level < len(ds):
                            level_parts.append(ds[level])
                        else:
                            level_parts.append(
                                torch.zeros_like(all_inputs_embeds[0]))
                    deepstack_embeds.append(torch.cat(level_parts, dim=0))
                return inputs_embeds, deepstack_embeds

        return inputs_embeds, None

    @torch.inference_mode()
    def generate(self, prompts, sampling_params, collect_logits: bool = False,
                 images=None, videos=None, use_tqdm: bool = False):
        """Generate completions for a batch of prompts.

        Uses unified chunked-prefill scheduling: every GPU step processes
        both decode tokens (for running seqs) and prefill chunks (for
        new/continuing seqs) in a single forward pass, matching vLLM's
        approach.
        """
        if isinstance(sampling_params, list):
            sp_list = sampling_params
        else:
            sp_list = [sampling_params] * len(prompts)

        seed = sp_list[0].seed
        if seed is not None:
            self._set_seeds(seed)

        eos = self.tokenizer.eos_token_id
        waiting: deque[Sequence] = deque()
        running: deque[Sequence] = deque()
        prefilling: deque[Sequence] = deque()

        seq_logits: dict[int, list[torch.Tensor]] = {}

        # Handle multimodal inputs
        if images is None:
            images = [None] * len(prompts)
        if videos is None:
            videos = [None] * len(prompts)

        _preprocess_t0 = time.perf_counter()

        def _make_seq(i):
            prompt = prompts[i]
            sp = sp_list[i]
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

                model = self.model_runner.model
                merge_size = model.config.vision.spatial_merge_size
                image_offsets = []
                video_offsets = []
                img_idx = 0
                vid_idx = 0
                i_tok = 0
                while i_tok < len(ids):
                    tid = ids[i_tok]
                    if tid == self.model_runner.config.image_token_id and seq.image_grid_thw and img_idx < len(seq.image_grid_thw):
                        image_offsets.append(i_tok)
                        t, h, w = seq.image_grid_thw[img_idx]
                        num_tokens = t * (h // merge_size) * (w // merge_size)
                        i_tok += num_tokens
                        img_idx += 1
                    elif tid == self.model_runner.config.video_token_id and seq.video_grid_thw and vid_idx < len(seq.video_grid_thw):
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
                seq.mrope_positions = torch.arange(len(ids), dtype=torch.int64).unsqueeze(0).expand(3, -1)
                seq.mrope_position_delta = 0
            else:
                ids = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
                seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
            return seq

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            all_seqs_ordered = list(pool.map(_make_seq, range(len(prompts))))

        for seq in all_seqs_ordered:
            waiting.append(seq)
            if collect_logits:
                seq_logits[id(seq)] = []

        all_seqs = list(waiting)
        num_prompts = len(prompts)
        _preprocess_time = time.perf_counter() - _preprocess_t0
        if os.environ.get("KB_NANO_STEP_PROFILE") == "1":
            print(f"[Profile] Preprocessing {num_prompts} seqs: {_preprocess_time:.3f}s")

        pbar = None
        if use_tqdm:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=num_prompts, desc="Processed prompts",
                         dynamic_ncols=True,
                         postfix="est. speed input: 0.00 toks/s, "
                                 "output: 0.00 toks/s")
        num_finished = 0
        total_in_toks = 0
        total_out_toks = 0

        use_greedy = (sp_list[0].temperature == 0.0
                      and not collect_logits)
        block_size = BLOCK_SIZE
        bm = self.block_manager
        num_blocks = bm._num_blocks
        watermark_blocks = max(int(num_blocks * 0.01), 1)

        _pbar_pending = 0
        _pbar_pending_in = 0
        _pbar_pending_out = 0

        step_profile = {
            "pure_decode": 0, "decode_tokens": 0, "decode_time": 0.0,
            "pure_mm_prefill": 0, "mm_prefill_tokens": 0, "mm_prefill_time": 0.0,
            "pure_text_prefill": 0, "text_prefill_tokens": 0,
            "mixed_mm": 0, "mixed_mm_pf_tokens": 0, "mixed_mm_dc_tokens": 0, "mixed_mm_time": 0.0,
            "mixed_text": 0, "mixed_text_pf_tokens": 0, "mixed_text_dc_tokens": 0,
        }
        _step_profile_active = os.environ.get("KB_NANO_STEP_PROFILE") == "1"

        def _finish_seq(seq: Sequence) -> None:
            nonlocal _pbar_pending, _pbar_pending_in, _pbar_pending_out
            seq.status = SeqStatus.FINISHED
            bm.deallocate(seq)
            self.encoder_cache.pop(id(seq), None)
            if pbar is not None:
                _pbar_pending += 1
                _pbar_pending_in += seq.num_prompt_tokens
                _pbar_pending_out += len(seq.generated_ids)

        def _flush_pbar() -> None:
            nonlocal _pbar_pending, _pbar_pending_in, _pbar_pending_out
            nonlocal total_in_toks, total_out_toks
            if _pbar_pending == 0:
                return
            total_in_toks += _pbar_pending_in
            total_out_toks += _pbar_pending_out
            elapsed = pbar.format_dict["elapsed"]
            if elapsed > 0:
                pbar.postfix = (
                    f"est. speed input: {total_in_toks / elapsed:.2f}"
                    f" toks/s, output: "
                    f"{total_out_toks / elapsed:.2f} toks/s")
            pbar.update(_pbar_pending)
            _pbar_pending = 0
            _pbar_pending_in = 0
            _pbar_pending_out = 0

        while waiting or running or prefilling:
            if pbar is not None:
                _flush_pbar()
            # =============================================================
            # FAST PATH: pure decode (most common steady-state)
            # No waiting/prefilling seqs, so skip the full scheduler.
            # =============================================================
            if running and not waiting and not prefilling and use_greedy:
                need_blocks = 0
                for seq in running:
                    if len(seq) % block_size == 1:
                        need_blocks += 1
                if need_blocks <= len(bm.free_block_ids):
                    if _step_profile_active:
                        _spt0 = time.perf_counter()
                        step_profile["fast_decode"] = step_profile.get("fast_decode", 0) + 1
                        step_profile["fast_decode_tokens"] = step_profile.get("fast_decode_tokens", 0) + len(running)
                    if _PROFILE:
                        _fp_t0 = time.perf_counter()
                    decode_seqs = list(running)
                    for seq in decode_seqs:
                        if len(seq) % block_size == 1:
                            seq.block_table.append(bm.free_block_ids.popleft())

                    mr = self.model_runner
                    n_dc = len(decode_seqs)
                    decode_data = mr._prepare_decode_arrays(decode_seqs)
                    if _PROFILE:
                        _fp_t1 = time.perf_counter()
                    if mr.world_size > 1:
                        mr._write_decode_shm(*decode_data)
                        mr.shm.buf[mr._SHM_FLAG_OFFSET] = 1
                        mr._signal_workers()
                    gpu_result = mr.run_decode_greedy_fast(decode_data)
                    if _PROFILE:
                        _fp_t2 = time.perf_counter()
                    if gpu_result is not None:
                        token_ids = gpu_result.tolist()
                        if _PROFILE:
                            _fp_t3 = time.perf_counter()
                        any_finished = False
                        for seq, tid in zip(decode_seqs, token_ids):
                            seq.append_token(tid)
                            done = len(seq.generated_ids) >= seq.max_tokens
                            if not seq.ignore_eos:
                                done = done or tid == eos
                            if done:
                                _finish_seq(seq)
                                any_finished = True
                        if any_finished:
                            running = deque(s for s in running
                                            if s.status != SeqStatus.FINISHED)
                        if _PROFILE:
                            _fp_t4 = time.perf_counter()
                            _fp = getattr(self, '_fast_path_profile', None)
                            if _fp is None:
                                _fp = {'prep': 0., 'gpu': 0., 'tolist': 0.,
                                       'post': 0., 'n': 0}
                                self._fast_path_profile = _fp
                            _fp['prep'] += _fp_t1 - _fp_t0
                            _fp['gpu'] += _fp_t2 - _fp_t1
                            _fp['tolist'] += _fp_t3 - _fp_t2
                            _fp['post'] += _fp_t4 - _fp_t3
                            _fp['n'] += 1

                        use_incr = True
                        while running and not waiting and not prefilling:
                            if any_finished:
                                decode_seqs = list(running)
                                n_dc = len(decode_seqs)
                                any_finished = False
                                use_incr = False

                            need_blocks = 0
                            for seq in decode_seqs:
                                if len(seq) % block_size == 1:
                                    need_blocks += 1
                            if need_blocks > len(bm.free_block_ids):
                                break
                            if os.environ.get("KB_NANO_STEP_PROFILE") == "1":
                                step_profile["fast_decode"] = step_profile.get("fast_decode", 0) + 1
                                step_profile["fast_decode_tokens"] = step_profile.get("fast_decode_tokens", 0) + n_dc
                            for seq in decode_seqs:
                                if len(seq) % block_size == 1:
                                    seq.block_table.append(
                                        bm.free_block_ids.popleft())

                            if _PROFILE:
                                _fp_t0 = time.perf_counter()
                            if use_incr:
                                decode_data = \
                                    mr._update_decode_arrays_incremental(
                                        n_dc, token_ids, decode_seqs)
                            else:
                                decode_data = mr._prepare_decode_arrays(
                                    decode_seqs)
                                use_incr = True
                            if _PROFILE:
                                _fp_t1 = time.perf_counter()
                            if mr.world_size > 1:
                                mr._write_decode_shm(*decode_data)
                                mr.shm.buf[mr._SHM_FLAG_OFFSET] = 1
                                mr._signal_workers()
                            gpu_result = mr.run_decode_greedy_fast(decode_data)
                            if _PROFILE:
                                _fp_t2 = time.perf_counter()
                            if gpu_result is not None:
                                token_ids = gpu_result.tolist()
                                if _PROFILE:
                                    _fp_t3 = time.perf_counter()
                                for seq, tid in zip(decode_seqs, token_ids):
                                    seq.append_token(tid)
                                    done = (len(seq.generated_ids)
                                            >= seq.max_tokens)
                                    if not seq.ignore_eos:
                                        done = done or tid == eos
                                    if done:
                                        _finish_seq(seq)
                                        any_finished = True
                                if any_finished:
                                    running = deque(
                                        s for s in running
                                        if s.status != SeqStatus.FINISHED)
                                if _PROFILE:
                                    _fp_t4 = time.perf_counter()
                                    _fp['prep'] += _fp_t1 - _fp_t0
                                    _fp['gpu'] += _fp_t2 - _fp_t1
                                    _fp['tolist'] += _fp_t3 - _fp_t2
                                    _fp['post'] += _fp_t4 - _fp_t3
                                    _fp['n'] += 1
                    if _step_profile_active:
                        step_profile["decode_time"] += time.perf_counter() - _spt0
                    continue

            elif running and not waiting and not prefilling:
                decode_seqs = list(running)
                need_blocks = 0
                for seq in decode_seqs:
                    if len(seq) % block_size == 1:
                        need_blocks += 1
                if need_blocks <= len(bm.free_block_ids):
                    for seq in decode_seqs:
                        if len(seq) % block_size == 1:
                            seq.block_table.append(bm.free_block_ids.popleft())
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
                                _finish_seq(seq)
                                finished_set.add(id(seq))
                        if finished_set:
                            running = deque(s for s in running if id(s) not in finished_set)
                    continue

            # =============================================================
            # SCHEDULE: one unified step
            # =============================================================
            token_budget = self.max_num_batched_tokens

            # --- 1. Allocate blocks for decode seqs that need a new block ---
            decode_seqs: list[Sequence] = []
            new_running: deque[Sequence] = deque()
            while running:
                seq = running.popleft()
                if len(decode_seqs) >= self.max_num_seqs:
                    new_running.append(seq)
                    continue
                needs_block = (len(seq) % block_size == 1)
                if needs_block:
                    if not bm.free_block_ids:
                        bm.deallocate(seq)
                        seq.preempt()
                        waiting.appendleft(seq)
                        continue
                    seq.block_table.append(bm.free_block_ids.popleft())
                decode_seqs.append(seq)
            running = new_running
            token_budget -= len(decode_seqs)

            # --- 2. Continue prefilling seqs already mid-prefill ---
            prefill_seqs: list[Sequence] = []
            prefill_chunk_sizes: list[int] = []
            still_prefilling: deque[Sequence] = deque()
            while prefilling and token_budget > 0:
                seq = prefilling.popleft()
                remaining = seq.num_remaining_prefill
                chunk = min(remaining, token_budget)
                blocks_needed = seq.blocks_needed_for(chunk)
                if blocks_needed > 0:
                    if len(bm.free_block_ids) < blocks_needed:
                        still_prefilling.append(seq)
                        continue
                    bm.allocate_n(seq, blocks_needed)
                prefill_seqs.append(seq)
                prefill_chunk_sizes.append(chunk)
                token_budget -= chunk
            while prefilling:
                still_prefilling.append(prefilling.popleft())
            prefilling = still_prefilling

            # --- 3. Admit new seqs from waiting queue ---
            total_peak = 0
            for seq in decode_seqs:
                total_peak += (seq.num_prompt_tokens + seq.max_tokens
                               + block_size - 1) // block_size
            for seq in running:
                total_peak += (seq.num_prompt_tokens + seq.max_tokens
                               + block_size - 1) // block_size
            for seq in prefilling:
                total_peak += (seq.num_prompt_tokens + seq.max_tokens
                               + block_size - 1) // block_size
            encoder_budget = self.max_num_batched_tokens
            while waiting and token_budget > 0:
                seq = waiting[0]
                prompt_len = seq.num_prompt_tokens

                has_mm = self.is_qwen_vl and (
                    getattr(seq, 'pixel_values', None) is not None
                    or getattr(seq, 'video_pixel_values', None) is not None)

                if has_mm:
                    chunk = min(prompt_len, token_budget)
                    if chunk > encoder_budget:
                        break
                else:
                    chunk = min(prompt_len, token_budget)

                blocks_needed = (chunk + block_size - 1) // block_size
                free = len(bm.free_block_ids)
                if free < blocks_needed + watermark_blocks:
                    break
                seq_peak = (prompt_len + seq.max_tokens
                            + block_size - 1) // block_size
                if total_peak + seq_peak > num_blocks:
                    break
                if len(prefill_seqs) + len(decode_seqs) >= self.max_num_seqs:
                    break
                waiting.popleft()
                bm.allocate_n(seq, blocks_needed)
                seq.status = SeqStatus.PREFILLING
                prefill_seqs.append(seq)
                prefill_chunk_sizes.append(chunk)
                token_budget -= chunk
                total_peak += seq_peak
                if has_mm:
                    encoder_budget -= chunk

            if not decode_seqs and not prefill_seqs:
                continue

            # =============================================================
            # EXECUTE: single forward pass
            # =============================================================
            n_pf = len(prefill_seqs)
            n_dc = len(decode_seqs)

            if _step_profile_active:
                _spt0 = time.perf_counter()
                _sp = step_profile
                has_mm_step = self.is_qwen_vl and any(
                    s.pixel_values is not None or s.video_pixel_values is not None
                    for s in prefill_seqs
                )
                _sp_cat = None
                if n_pf == 0:
                    _sp["pure_decode"] += 1
                    _sp["decode_tokens"] += n_dc
                    _sp_cat = "decode_time"
                elif n_dc == 0 and has_mm_step:
                    _sp["pure_mm_prefill"] += 1
                    _sp["mm_prefill_tokens"] += sum(prefill_chunk_sizes)
                    _sp_cat = "mm_prefill_time"
                elif n_dc == 0:
                    _sp["pure_text_prefill"] += 1
                    _sp["text_prefill_tokens"] += sum(prefill_chunk_sizes)
                elif has_mm_step:
                    _sp["mixed_mm"] += 1
                    _sp["mixed_mm_pf_tokens"] += sum(prefill_chunk_sizes)
                    _sp["mixed_mm_dc_tokens"] += n_dc
                    _sp_cat = "mixed_mm_time"
                else:
                    _sp["mixed_text"] += 1
                    _sp["mixed_text_pf_tokens"] += sum(prefill_chunk_sizes)
                    _sp["mixed_text_dc_tokens"] += n_dc

            if n_pf == 0 and use_greedy:
                # Pure decode with CUDA graphs (fast path)
                gpu_result = self.model_runner.call_decode_greedy(decode_seqs)
                if gpu_result is not None:
                    token_ids = gpu_result.tolist()
                    finished_set = set()
                    for seq, tid in zip(decode_seqs, token_ids):
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            _finish_seq(seq)
                            finished_set.add(id(seq))
                        else:
                            running.append(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)
                else:
                    running.extend(decode_seqs)
            elif n_pf == 0:
                # Pure decode, non-greedy
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
                            _finish_seq(seq)
                            finished_set.add(id(seq))
                        else:
                            running.append(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)
                else:
                    running.extend(decode_seqs)
            elif n_dc == 0:
                # Pure prefill (no running decode seqs)
                has_mm = self.is_qwen_vl and any(
                    s.pixel_values is not None or s.video_pixel_values is not None
                    for s in prefill_seqs
                )
                if has_mm:
                    inputs_embeds, deepstack_embeds = self._run_vision_encoder(
                        prefill_seqs, chunk_sizes=prefill_chunk_sizes)
                    input_ids_t, positions_t = self.model_runner.prepare_mixed_batch(
                        prefill_seqs, prefill_chunk_sizes, [],
                    )
                    logits = self.model_runner.run_model(
                        input_ids_t, positions_t, True,
                        inputs_embeds=inputs_embeds,
                        deepstack_embeds=deepstack_embeds,
                    )
                    reset_context()
                    if _step_profile_active:
                        torch.cuda.synchronize()
                        step_profile["mm_prefill_time"] += time.perf_counter() - _spt0
                else:
                    logits = self.model_runner.call(
                        "run_mixed", prefill_seqs, prefill_chunk_sizes, [],
                    )
                if logits is not None:
                    self._process_prefill_logits(
                        logits, prefill_seqs, prefill_chunk_sizes,
                        sp_list[0], eos, collect_logits, seq_logits,
                        running, prefilling, bm, block_size,
                        finish_seq=_finish_seq,
                    )
            else:
                # Mixed batch: prefill + decode together
                has_mm = self.is_qwen_vl and any(
                    s.pixel_values is not None or s.video_pixel_values is not None
                    for s in prefill_seqs
                )
                if has_mm:
                    pf_embeds, deepstack_embeds = self._run_vision_encoder(
                        prefill_seqs, chunk_sizes=prefill_chunk_sizes)
                    embed_fn = self.model_runner.model.get_input_embeddings()
                    dc_ids = torch.tensor(
                        [s.last_token for s in decode_seqs],
                        dtype=torch.int64, device="cuda",
                    )
                    dc_embeds = embed_fn(dc_ids)
                    inputs_embeds = torch.cat([pf_embeds, dc_embeds], dim=0)
                    if deepstack_embeds is not None:
                        dc_zeros = torch.zeros_like(dc_embeds).unsqueeze(0).expand(
                            len(deepstack_embeds), -1, -1)
                        deepstack_embeds = [
                            torch.cat([ds, dc_zeros[i]], dim=0)
                            for i, ds in enumerate(deepstack_embeds)
                        ]
                    input_ids_t, positions_t = self.model_runner.prepare_mixed_batch(
                        prefill_seqs, prefill_chunk_sizes, decode_seqs,
                    )
                    logits = self.model_runner.run_model(
                        input_ids_t, positions_t, True,
                        inputs_embeds=inputs_embeds,
                        deepstack_embeds=deepstack_embeds,
                    )
                    reset_context()
                    if _step_profile_active:
                        torch.cuda.synchronize()
                        step_profile["mixed_mm_time"] += time.perf_counter() - _spt0
                else:
                    logits = self.model_runner.call(
                        "run_mixed", prefill_seqs, prefill_chunk_sizes, decode_seqs,
                    )
                if logits is not None:
                    pf_logits = logits[:n_pf]
                    dc_logits = logits[n_pf:]

                    self._process_prefill_logits(
                        pf_logits, prefill_seqs, prefill_chunk_sizes,
                        sp_list[0], eos, collect_logits, seq_logits,
                        running, prefilling, bm, block_size,
                        finish_seq=_finish_seq,
                    )

                    if collect_logits:
                        for i, seq in enumerate(decode_seqs):
                            seq_logits[id(seq)].append(dc_logits[i:i+1].cpu())
                    dc_token_ids = self._sample(dc_logits, sp_list[0])
                    finished_set = set()
                    for seq, tid in zip(decode_seqs, dc_token_ids):
                        seq.append_token(tid)
                        done = len(seq.generated_ids) >= seq.max_tokens
                        if not seq.ignore_eos:
                            done = done or tid == eos
                        if done:
                            _finish_seq(seq)
                            finished_set.add(id(seq))
                        else:
                            running.append(seq)
                    if finished_set:
                        running = deque(s for s in running if id(s) not in finished_set)
                else:
                    running.extend(decode_seqs)

        if pbar is not None:
            _flush_pbar()
            pbar.close()

        if _step_profile_active:
            sp = step_profile
            fd = sp.get("fast_decode", 0)
            fdt = sp.get("fast_decode_tokens", 0)
            slow = (sp["pure_decode"] + sp["pure_mm_prefill"]
                    + sp["pure_text_prefill"] + sp["mixed_mm"] + sp["mixed_text"])
            total = fd + slow
            print(f"\n[Step Profile] total_steps={total} (fast_decode={fd}, scheduled={slow})")
            print(f"  fast_decode:       {fd:6d} steps, {fdt:10d} tokens, {sp['decode_time']:.3f}s")
            print(f"  pure_decode:       {sp['pure_decode']:6d} steps, {sp['decode_tokens']:10d} tokens")
            print(f"  pure_text_prefill: {sp['pure_text_prefill']:6d} steps, {sp['text_prefill_tokens']:10d} tokens")
            print(f"  pure_mm_prefill:   {sp['pure_mm_prefill']:6d} steps, {sp['mm_prefill_tokens']:10d} tokens, {sp['mm_prefill_time']:.3f}s")
            print(f"  mixed_text:        {sp['mixed_text']:6d} steps, pf={sp['mixed_text_pf_tokens']:10d} dc={sp['mixed_text_dc_tokens']:10d}")
            print(f"  mixed_mm:          {sp['mixed_mm']:6d} steps, pf={sp['mixed_mm_pf_tokens']:10d} dc={sp['mixed_mm_dc_tokens']:10d}, {sp['mixed_mm_time']:.3f}s")

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

    def _process_prefill_logits(
        self, logits, prefill_seqs, prefill_chunk_sizes,
        sp, eos, collect_logits, seq_logits,
        running, prefilling, bm, block_size,
        finish_seq=None,
    ):
        """Handle output from prefill sequences after a forward pass.

        For sequences whose prefill is complete, sample the first decode
        token. For sequences still mid-prefill, update num_computed_tokens
        and move them to the prefilling queue.
        """
        # Separate seqs into "done prefilling" vs "still prefilling"
        sample_seqs = []
        sample_logits = []
        for i, (seq, chunk) in enumerate(zip(prefill_seqs, prefill_chunk_sizes)):
            seq.num_computed_tokens += chunk
            if seq.num_remaining_prefill == 0:
                # Prefill complete — sample first decode token
                sample_seqs.append(seq)
                sample_logits.append(logits[i:i+1])
            else:
                # More prefill chunks needed
                prefilling.append(seq)

        if not sample_seqs:
            return

        sample_logits_t = torch.cat(sample_logits, dim=0)
        if collect_logits:
            for i, seq in enumerate(sample_seqs):
                seq_logits[id(seq)].append(sample_logits_t[i:i+1].cpu())
        token_ids = self._sample(sample_logits_t, sp)
        for seq, tid in zip(sample_seqs, token_ids):
            seq.append_token(tid)
            seq.status = SeqStatus.RUNNING
            done = len(seq.generated_ids) >= seq.max_tokens
            if not seq.ignore_eos:
                done = done or tid == eos
            if done:
                if finish_seq is not None:
                    finish_seq(seq)
                else:
                    seq.status = SeqStatus.FINISHED
                    bm.deallocate(seq)
            else:
                running.append(seq)
