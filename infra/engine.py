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
        self._cached_embeds = None  # cached vision encoder output (full seq embeddings)
        self._cached_deepstack = None  # cached deepstack embeddings

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
        self.mrope_positions = None
        self.mrope_position_delta = 0
        self._cached_embeds = None
        self._cached_deepstack = None


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
                 max_num_batched_tokens: int | None = None):
        self.rank = rank
        self.world_size = world_size
        self.enforce_eager = enforce_eager
        self.event = event
        self.block_size = BLOCK_SIZE
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = ((max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 2) * BLOCK_SIZE
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else _DEFAULT_MAX_NUM_SEQS
        self.max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else _DEFAULT_MAX_NUM_BATCHED_TOKENS

        torch.cuda.set_device(rank)
        from datetime import timedelta
        dist.init_process_group(
            "nccl", f"tcp://localhost:{NCCL_PORT}",
            world_size=world_size, rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
            timeout=timedelta(hours=1),
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

        self.dtype = dtype
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")

        self.model, self.config = load_model(
            model_name, torch.device(f"cuda:{rank}"), dtype,
        )
        self.is_qwen_vl = hasattr(self.config, "mrope_section")
        self._share_trtllm_workspace()
        self._share_activation_buffers()
        self.warmup_model()
        self._warmup_deep_gemm()
        self._apply_torch_compile()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        self._init_greedy_buffers()
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
        """Share activation buffers across all layers.

        Layers execute sequentially so buffers are safe to reuse.
        Shares SiluAndMul activation buffers, FusedExperts MoE
        intermediate caches, and FP8 quant/GEMM output buffers
        to avoid per-layer allocation.
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
            FusedExperts._use_shared_cache = True

        self._share_fp8_buffers()

    def _share_fp8_buffers(self):
        """Share FP8 quantization and GEMM output buffers across layers.

        All FP8Linear instances in the model share a single set of quant
        buffers (q_buf, s_buf) and a single GEMM output buffer since
        decoder layers execute sequentially.
        """
        from ..tasks.baseline.L1.fp8_linear import FP8Linear
        fp8_linears = [
            m for m in self.model.modules() if isinstance(m, FP8Linear)
        ]
        if len(fp8_linears) <= 1:
            return

        max_elements = self.max_num_batched_tokens * self.config.hidden_size
        inter_size = getattr(self.config, "intermediate_size", self.config.hidden_size)
        max_elements = max(max_elements, self.max_num_batched_tokens * inter_size * 2)
        group_size = fp8_linears[0].block_size[1]
        device = f"cuda:{self.rank}"

        _FP8_DTYPE = torch.float8_e4m3fn
        shared_q_buf = torch.empty(max_elements, device=device, dtype=_FP8_DTYPE)
        shared_s_buf = torch.empty(
            max_elements // group_size, device=device, dtype=torch.float32,
        )
        for m in fp8_linears:
            m.quant.set_shared_buffers(shared_q_buf, shared_s_buf)

        num_heads = self.config.num_attention_heads // self.world_size
        num_kv_heads = self.config.num_key_value_heads // self.world_size
        head_dim = self.config.head_dim
        qkv_n = (num_heads + 2 * num_kv_heads) * head_dim
        gate_up_n = inter_size * 2 // self.world_size
        max_n = max(qkv_n, gate_up_n, self.config.hidden_size)
        max_m = self.max_num_batched_tokens
        shared_out_buf = torch.empty(
            max_m * max_n, device=device, dtype=torch.bfloat16,
        )
        for m in fp8_linears:
            m.mm.set_shared_buffer(shared_out_buf)

        # Also share quant buffers in FusedExperts FP8 path
        from ..tasks.baseline.L2.fused_experts import FusedExperts
        for fe in self.model.modules():
            if isinstance(fe, FusedExperts) and hasattr(fe, 'quant'):
                fe.quant.set_shared_buffers(shared_q_buf, shared_s_buf)

        # Share FP8 output buffers for fused RMSNorm+FP8 modules.
        # Safe to reuse shared_q_buf/shared_s_buf since the fused norm's
        # FP8 output is consumed by the next GEMM before any other write.
        from ..tasks.baseline.L1.fused_rmsnorm_fp8 import FusedRMSNormFP8Quant
        for m in self.model.modules():
            if isinstance(m, FusedRMSNormFP8Quant):
                m.set_shared_buffers(shared_q_buf, shared_s_buf)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        warmup_len = min(self.max_model_len, self.max_num_batched_tokens)
        num_seqs = min(self.max_num_batched_tokens // warmup_len, self.max_num_seqs)
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]

        # Disable fused norm during warmup to avoid buffer state issues
        from ..tasks.baseline.L3.llama_decoder import LlamaDecoderLayer
        fused_layers = []
        for m in self.model.modules():
            if isinstance(m, LlamaDecoderLayer) and m._use_fused_norm_fp8:
                fused_layers.append(m)
                m._use_fused_norm_fp8 = False

        self.run(seqs, True)

        for m in fused_layers:
            m._use_fused_norm_fp8 = True

        torch.cuda.empty_cache()

    def _warmup_deep_gemm(self):
        """Pre-JIT DeepGEMM kernels for all (M, N, K) shapes used during inference.

        DeepGEMM compiles CUDA kernels on first invocation per unique shape.
        Without warmup, JIT compilation stalls the pipeline mid-inference.
        """
        try:
            import deep_gemm as dg
            if not hasattr(dg, "fp8_gemm_nt"):
                return
        except ImportError:
            return

        from ..tasks.baseline.L1.fp8_linear import FP8Linear
        fp8_linears = [m for m in self.model.modules() if isinstance(m, FP8Linear)]
        if not fp8_linears:
            return

        seen_shapes: set[tuple[int, int]] = set()
        weight_info: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        for parent in self.model.modules():
            if not hasattr(parent, 'linear_op'):
                continue
            if not isinstance(parent.linear_op, FP8Linear):
                continue
            w = parent.weight
            if w.dtype != torch.float8_e4m3fn:
                continue
            N, K = w.shape
            if N % 64 != 0 or K % 128 != 0:
                continue
            key = (N, K)
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            ws = parent.weight_scale_inv
            weight_info.append((N, K, w, ws))

        if not weight_info:
            return

        max_tokens = self.max_num_batched_tokens
        device = f"cuda:{self.rank}"
        num_sms = torch.cuda.get_device_properties(self.rank).multi_processor_count

        m_values = set()
        m_values.update([1, 2, 4] + list(range(8, 65, 8)))
        block_ms = [64, 128, 256]
        all_ns = set(N for N, K, _, _ in weight_info)
        for N in all_ns:
            block_ns = list(range(16, min(257, N + 1), 16))
            for block_m in block_ms:
                for block_n in block_ns:
                    for wave in range(1, 11):
                        target_blocks = wave * num_sms
                        cdiv_n = (N + block_n - 1) // block_n
                        m = target_blocks * block_m // cdiv_n if cdiv_n else 0
                        if 1 <= m <= max_tokens:
                            m_values.add(m)
                    for multiple in range(1, max_tokens // block_m + 1):
                        m = multiple * block_m
                        if m <= max_tokens:
                            m_values.add(m)

        sorted_m = sorted(m for m in m_values if m <= max_tokens)
        total = len(sorted_m) * len(weight_info)
        if self.rank == 0:
            print(f"  DeepGEMM warmup: {total} kernels "
                  f"({len(sorted_m)} M-values x {len(weight_info)} weight shapes)")

        e8m0 = torch.cuda.get_device_capability(self.rank)[0] >= 10
        dg_kwargs = {} if e8m0 else {"disable_ue8m0_cast": True}

        for N, K, w, ws in weight_info:
            a = torch.empty(max_tokens, K, device=device, dtype=torch.float8_e4m3fn)
            a_s = torch.empty(max_tokens, K // 128, device=device, dtype=torch.float32)
            out = torch.empty(max_tokens, N, device=device, dtype=torch.bfloat16)
            for m in sorted_m:
                dg.fp8_gemm_nt(
                    (a[:m], a_s[:m]),
                    (w, ws),
                    out[:m],
                    **dg_kwargs,
                )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _apply_torch_compile(self):
        """Apply torch.compile to vision encoder and decoder layers.

        Vision encoder: compiled with dynamic=True since vision token counts
        vary per image. Reduces kernel launch overhead for the 27 vision blocks.

        Decoder layers: compiled when Inductor XBLOCK limits allow it (not all
        GPU architectures support the generated block sizes). Falls back to
        eager mode gracefully.
        """
        if self.enforce_eager:
            return
        if os.environ.get("KB_NANO_NO_COMPILE", "0") == "1":
            return

        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 128

        try:
            from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
            for k in TRITON_MAX_BLOCK:
                TRITON_MAX_BLOCK[k] = max(TRITON_MAX_BLOCK[k], 2**16)
        except ImportError:
            pass

        compile_decoder = os.environ.get("KB_NANO_COMPILE_DECODER", "0") == "1"

        if hasattr(self.model, 'visual') and hasattr(self.model.visual, '_run_blocks'):
            if self.rank == 0:
                print("  torch.compile: vision encoder blocks")
            self.model.visual._run_blocks = torch.compile(
                self.model.visual._run_blocks, dynamic=True,
            )

        if compile_decoder:
            model_inner = getattr(self.model, 'model', None)
            if model_inner is not None and hasattr(model_inner, 'layers'):
                n_layers = len(model_inner.layers)
                if self.rank == 0:
                    print(f"  torch.compile: {n_layers} decoder layers")
                for i in range(n_layers):
                    model_inner.layers[i] = torch.compile(
                        model_inner.layers[i], dynamic=False,
                    )

        torch.cuda.synchronize()

    def allocate_kv_cache(self):
        if not hasattr(self, '_attn_layers') or not self._attn_layers:
            self._attn_layers = []
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    self._attn_layers.append(module)

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = self.config.num_key_value_heads // self.world_size
        head_dim = self.config.head_dim
        num_layers = self.config.num_hidden_layers
        elem_size = torch.finfo(torch.get_default_dtype()).bits // 8
        block_bytes = 2 * num_layers * BLOCK_SIZE * num_kv_heads * head_dim * elem_size
        num_blocks = int(total * self.gpu_memory_utilization - used - peak + current) // block_bytes
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
        use_mrope = self.is_qwen_vl
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
            base_pos = len(seq) - 1
            if use_mrope:
                decode_pos = base_pos + seq.mrope_position_delta
                mrope_pos_list.append(
                    torch.full((3, 1), decode_pos, dtype=torch.int64)
                )
            else:
                positions.append(base_pos)
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
        return (input_ids_t, positions_t)

    @torch.inference_mode()
    def run_model(self, input_ids, positions, is_prefill, inputs_embeds=None,
                  deepstack_embeds=None):
        if is_prefill or self.enforce_eager or not self.graph_bs_list or input_ids.size(0) > self.graph_bs_list[-1]:
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

        if self.enforce_eager or not self.graph_bs_list or n > self.graph_bs_list[-1]:
            return self._run_decode_greedy_eager(n, ids_np, pos_np, sm_np, cl_np, bt_np)

        self._run_graph_from_numpy(n, ids_np, pos_np, sm_np, cl_np, bt_np)
        return self._greedy_from_hidden(n)

    def _run_graph_from_numpy(self, n, ids_np, pos_np, sm_np, cl_np, bt_np):
        """Copy numpy arrays into graph vars and replay the CUDA graph."""
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

    @torch.inference_mode()
    def run_decode_greedy_fast_async(self, decode_data):
        """Like run_decode_greedy_fast but starts async D2H copy.

        Returns (has_result, n) -- caller must call _wait_async_tokens(n)
        later to get the Python list of token IDs.
        """
        n, ids_np, pos_np, sm_np, cl_np, bt_np = decode_data

        if self.enforce_eager or not self.graph_bs_list or n > self.graph_bs_list[-1]:
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
        block_tables = self._eager_block_tables[:n, :bt_cols].contiguous()

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
        from ..tasks.baseline.L2.fused_experts import FusedExperts
        if FusedExperts._use_shared_cache and not FusedExperts._use_flashinfer:
            print("  MoE model detected without FlashInfer — skipping CUDA graph capture (using eager mode for decode)")
            self.enforce_eager = True
            self.graph_bs_list = []
            self.graphs = {}
            self.graph_pool = None
            self.graph_vars = None
            self._graph_bs_for_n = []
            return

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

        ar_ctx = self.custom_ar.capture() if self.custom_ar is not None else nullcontext()
        with ar_ctx:
            for bs in reversed(self.graph_bs_list):
                graph = torch.cuda.CUDAGraph()
                set_context(
                    False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                    max_context_len=self.max_model_len,
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
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
    ):
        self.model_name = model_name
        self.seed = seed
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else _DEFAULT_MAX_NUM_SEQS
        self.max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else _DEFAULT_MAX_NUM_BATCHED_TOKENS
        self._set_seeds(seed)

        # Unique shared memory name to avoid collisions
        shm_name = f"sllama_{uuid.uuid4().hex[:8]}"

        mr_kwargs = dict(
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )

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
                kwargs=mr_kwargs,
            )
            p.start()
            self.workers.append(p)
            self.events.append(event)

        # Rank 0 model runner (events is a list for rank 0)
        self.model_runner = ModelRunner(
            model_name, 0, tensor_parallel_size, dtype,
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

    def preprocess_multimodal(self, prompt, images=None, videos=None):
        """Public API: preprocess a multimodal prompt into a dict suitable for generate().

        Returns a dict with keys: token_ids, pixel_values, image_grid_thw,
        video_pixel_values, video_grid_thw, mrope_positions, mrope_position_delta.
        Pass the returned dict as an element of the prompts list to generate().
        """
        (ids, pixel_values, image_grid_thw,
         video_pv, video_grid_thw) = self._preprocess_multimodal(
            prompt, images=images, videos=videos,
        )
        image_grid_thw_list = image_grid_thw.tolist() if image_grid_thw is not None else None
        video_grid_thw_list = video_grid_thw.tolist() if video_grid_thw is not None else None

        model = self.model_runner.model
        merge_size = model.config.vision.spatial_merge_size
        image_offsets = []
        video_offsets = []
        img_idx = 0
        vid_idx = 0
        i_tok = 0
        while i_tok < len(ids):
            tid = ids[i_tok]
            if tid == self.model_runner.config.image_token_id and image_grid_thw_list and img_idx < len(image_grid_thw_list):
                image_offsets.append(i_tok)
                t, h, w = image_grid_thw_list[img_idx]
                num_tokens = t * (h // merge_size) * (w // merge_size)
                i_tok += num_tokens
                img_idx += 1
            elif tid == self.model_runner.config.video_token_id and video_grid_thw_list and vid_idx < len(video_grid_thw_list):
                video_offsets.append(i_tok)
                t, h, w = video_grid_thw_list[vid_idx]
                num_tokens = t * (h // merge_size) * (w // merge_size)
                i_tok += num_tokens
                vid_idx += 1
            else:
                i_tok += 1

        mrope_positions, delta = model.get_mrope_input_positions(
            ids,
            image_grid_thw=image_grid_thw_list,
            video_grid_thw=video_grid_thw_list,
            image_offsets=image_offsets if image_offsets else None,
            video_offsets=video_offsets if video_offsets else None,
        )
        return {
            "token_ids": ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw_list,
            "video_pixel_values": video_pv,
            "video_grid_thw": video_grid_thw_list,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": delta,
        }

    def preprocess_chat(self, messages):
        """Preprocess an OpenAI-format chat message list into a dict for generate().

        Extracts images/videos from the message content, applies the chat
        template through the HF processor, and computes M-RoPE positions.
        This matches the preprocessing path used by vLLM's llm.chat().
        """
        from PIL import Image
        import base64
        from io import BytesIO

        images = []
        videos = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            for c in msg.get("content", []):
                ctype = c.get("type", "")
                if ctype == "image_url":
                    url = c["image_url"]["url"]
                    if url.startswith("data:image/"):
                        b64 = url.split(",", 1)[1]
                        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                    else:
                        path = url.replace("file://", "")
                        img = Image.open(path).convert("RGB")
                    images.append(img)
                elif ctype == "video_url":
                    video_url = c["video_url"]["url"]
                    video_path = video_url
                    for prefix in ("file://", "https://", "http://"):
                        if video_path.startswith(prefix):
                            video_path = video_path[len(prefix):]
                            break
                    if not os.path.exists(video_path):
                        from huggingface_hub import hf_hub_download
                        parts = video_path.split("/")
                        try:
                            hf_idx = next(i for i, p in enumerate(parts) if p == "datasets")
                            repo_id = "/".join(parts[hf_idx + 1:hf_idx + 3])
                            resolve_idx = parts.index("resolve")
                            rel_path = "/".join(parts[resolve_idx + 2:])
                            video_path = hf_hub_download(
                                repo_id=repo_id, filename=rel_path, repo_type="dataset",
                            )
                        except (StopIteration, ValueError):
                            pass
                    import decord
                    decord.bridge.set_bridge("native")
                    vr = decord.VideoReader(video_path)
                    total = len(vr)
                    num_frames = min(total, 16)
                    indices = [int(i * total / num_frames) for i in range(num_frames)]
                    frames = [Image.fromarray(vr[idx].asnumpy()).convert("RGB")
                              for idx in indices]
                    videos.append(frames)

        content_parts = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            for c in msg.get("content", []):
                ctype = c.get("type", "")
                if ctype == "text":
                    content_parts.append({"type": "text", "text": c["text"]})
                elif ctype == "image_url":
                    content_parts.append({"type": "image", "image": images[len([
                        p for p in content_parts if p.get("type") == "image"
                    ])]})
                elif ctype == "video_url":
                    content_parts.append({"type": "video", "video": videos[len([
                        p for p in content_parts if p.get("type") == "video"
                    ])]})

        proc_messages = [{"role": "user", "content": content_parts}]
        text = self.processor.apply_chat_template(
            proc_messages, tokenize=False, add_generation_prompt=True,
        )
        proc_images = images if images else None
        proc_videos = videos if videos else None
        inputs = self.processor(
            text=[text], images=proc_images, videos=proc_videos,
            return_tensors="pt", padding=True,
        )
        ids = inputs["input_ids"][0].tolist()
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        video_pv = inputs.get("pixel_values_videos", None)
        video_grid_thw = inputs.get("video_grid_thw", None)

        image_grid_thw_list = image_grid_thw.tolist() if image_grid_thw is not None else None
        video_grid_thw_list = video_grid_thw.tolist() if video_grid_thw is not None else None

        model = self.model_runner.model
        merge_size = model.config.vision.spatial_merge_size
        image_offsets = []
        video_offsets = []
        img_idx = 0
        vid_idx = 0
        i_tok = 0
        while i_tok < len(ids):
            tid = ids[i_tok]
            if tid == self.model_runner.config.image_token_id and image_grid_thw_list and img_idx < len(image_grid_thw_list):
                image_offsets.append(i_tok)
                t, h, w = image_grid_thw_list[img_idx]
                num_tokens = t * (h // merge_size) * (w // merge_size)
                i_tok += num_tokens
                img_idx += 1
            elif tid == self.model_runner.config.video_token_id and video_grid_thw_list and vid_idx < len(video_grid_thw_list):
                video_offsets.append(i_tok)
                t, h, w = video_grid_thw_list[vid_idx]
                num_tokens = t * (h // merge_size) * (w // merge_size)
                i_tok += num_tokens
                vid_idx += 1
            else:
                i_tok += 1

        mrope_positions, delta = model.get_mrope_input_positions(
            ids,
            image_grid_thw=image_grid_thw_list,
            video_grid_thw=video_grid_thw_list,
            image_offsets=image_offsets if image_offsets else None,
            video_offsets=video_offsets if video_offsets else None,
        )
        return {
            "token_ids": ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw_list,
            "video_pixel_values": video_pv,
            "video_grid_thw": video_grid_thw_list,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": delta,
        }

    @torch.inference_mode()
    def _run_vision_encoder(self, seqs):
        """Run vision encoder for sequences with multimodal data and merge embeddings.

        Batches pixel values across sequences into single model.visual() calls
        (matching vLLM's approach) for better GPU utilization. Caches embeddings
        on the Sequence object so the vision encoder only runs once per sequence.

        Returns (inputs_embeds, deepstack_embeds) where deepstack_embeds is a list
        of tensors for Qwen3-VL DeepStack, or None for Qwen2-VL.
        """
        model = self.model_runner.model
        all_inputs_embeds = []
        has_deepstack = hasattr(model.visual, 'deepstack_merger_list')
        all_deepstack = [] if has_deepstack else None
        merge_size = model.config.vision.spatial_merge_size

        uncached = []
        seq_to_idx = {}
        for i, seq in enumerate(seqs):
            seq_to_idx[id(seq)] = i
            if seq._cached_embeds is not None:
                all_inputs_embeds.append(seq._cached_embeds)
                if has_deepstack:
                    all_deepstack.append(seq._cached_deepstack or [])
            else:
                uncached.append(seq)
                all_inputs_embeds.append(None)
                if has_deepstack:
                    all_deepstack.append(None)

        if not uncached:
            inputs_embeds = torch.cat(all_inputs_embeds, dim=0)
            if has_deepstack and all_deepstack:
                return inputs_embeds, self._merge_deepstack(all_deepstack, all_inputs_embeds)
            return inputs_embeds, None

        # Batch image pixel values across all uncached sequences
        img_pv_parts = []
        img_thw_parts = []
        img_sizes_per_seq = {}
        for seq in uncached:
            if seq.pixel_values is not None:
                pv = seq.pixel_values.cuda() if not seq.pixel_values.is_cuda else seq.pixel_values
                img_pv_parts.append(pv)
                thw = seq.image_grid_thw
                img_thw_parts.extend(thw)
                sizes = [t * (h // merge_size) * (w // merge_size) for t, h, w in thw]
                img_sizes_per_seq[id(seq)] = sizes

        batched_img_embeds = None
        batched_img_ds = None
        if img_pv_parts:
            cat_pv = torch.cat(img_pv_parts, dim=0)
            vis_out = model.visual(cat_pv, grid_thw=img_thw_parts)
            if has_deepstack:
                batched_img_embeds, batched_img_ds = vis_out
            else:
                batched_img_embeds = vis_out

        # Batch video pixel values across all uncached sequences
        vid_pv_parts = []
        vid_thw_parts = []
        vid_sizes_per_seq = {}
        for seq in uncached:
            if seq.video_pixel_values is not None:
                pv = seq.video_pixel_values.cuda() if not seq.video_pixel_values.is_cuda else seq.video_pixel_values
                vid_pv_parts.append(pv)
                thw = seq.video_grid_thw
                vid_thw_parts.extend(thw)
                sizes = [t * (h // merge_size) * (w // merge_size) for t, h, w in thw]
                vid_sizes_per_seq[id(seq)] = sizes

        batched_vid_embeds = None
        batched_vid_ds = None
        if vid_pv_parts:
            cat_pv = torch.cat(vid_pv_parts, dim=0)
            vis_out = model.visual(cat_pv, grid_thw=vid_thw_parts)
            if has_deepstack:
                batched_vid_embeds, batched_vid_ds = vis_out
            else:
                batched_vid_embeds = vis_out

        # Distribute batched embeddings back to individual sequences
        img_offset = 0
        vid_offset = 0
        for seq in uncached:
            token_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cuda")
            text_embeds = model.get_input_embeddings()(token_ids)
            seq_deepstack = [] if has_deepstack else None

            if seq.pixel_values is not None and id(seq) in img_sizes_per_seq:
                total_tokens = sum(img_sizes_per_seq[id(seq)])
                seq_img_embeds = batched_img_embeds[img_offset:img_offset + total_tokens]

                mask = token_ids == self.model_runner.config.image_token_id
                if mask.any():
                    text_embeds[mask] = seq_img_embeds.to(text_embeds.dtype)

                if has_deepstack and batched_img_ds:
                    for ds_level in batched_img_ds:
                        ds_expanded = torch.zeros_like(text_embeds)
                        if mask.any():
                            ds_expanded[mask] = ds_level[img_offset:img_offset + total_tokens].to(text_embeds.dtype)
                        seq_deepstack.append(ds_expanded)

                img_offset += total_tokens
                seq.pixel_values = None

            if seq.video_pixel_values is not None and id(seq) in vid_sizes_per_seq:
                total_tokens = sum(vid_sizes_per_seq[id(seq)])
                seq_vid_embeds = batched_vid_embeds[vid_offset:vid_offset + total_tokens]

                mask = token_ids == self.model_runner.config.video_token_id
                if mask.any():
                    text_embeds[mask] = seq_vid_embeds.to(text_embeds.dtype)

                if has_deepstack and batched_vid_ds:
                    for i, ds_level in enumerate(batched_vid_ds):
                        ds_expanded = torch.zeros_like(text_embeds)
                        if mask.any():
                            ds_expanded[mask] = ds_level[vid_offset:vid_offset + total_tokens].to(text_embeds.dtype)
                        if i < len(seq_deepstack):
                            seq_deepstack[i] = seq_deepstack[i] + ds_expanded
                        else:
                            seq_deepstack.append(ds_expanded)

                vid_offset += total_tokens
                seq.video_pixel_values = None

            seq._cached_embeds = text_embeds
            seq._cached_deepstack = seq_deepstack

            idx = seq_to_idx[id(seq)]
            all_inputs_embeds[idx] = text_embeds
            if has_deepstack:
                all_deepstack[idx] = seq_deepstack or []

        inputs_embeds = torch.cat(all_inputs_embeds, dim=0)

        if has_deepstack and all_deepstack:
            return inputs_embeds, self._merge_deepstack(all_deepstack, all_inputs_embeds)

        return inputs_embeds, None

    def _merge_deepstack(self, all_deepstack, all_inputs_embeds):
        """Merge per-sequence deepstack features into batched tensors."""
        num_levels = max((len(ds) for ds in all_deepstack if ds), default=0)
        if num_levels == 0:
            return None
        deepstack_embeds = []
        for level in range(num_levels):
            level_parts = []
            for ds, ie in zip(all_deepstack, all_inputs_embeds):
                if ds and level < len(ds):
                    level_parts.append(ds[level])
                else:
                    level_parts.append(torch.zeros_like(ie))
            deepstack_embeds.append(torch.cat(level_parts, dim=0))
        return deepstack_embeds

    def _slice_mm_embeds(self, prefill_seqs, prefill_chunk_sizes,
                         inputs_embeds, deepstack_embeds):
        """Slice full-sequence embeddings to match chunked prefill tokens."""
        parts = []
        offset = 0
        for seq, chunk_size in zip(prefill_seqs, prefill_chunk_sizes):
            seq_len = len(seq.token_ids)
            start = seq.num_computed_tokens
            parts.append(inputs_embeds[offset + start:offset + start + chunk_size])
            offset += seq_len
        sliced = torch.cat(parts, dim=0)

        sliced_ds = None
        if deepstack_embeds is not None:
            sliced_ds = []
            for level_embed in deepstack_embeds:
                level_parts = []
                off = 0
                for seq, chunk_size in zip(prefill_seqs, prefill_chunk_sizes):
                    seq_len = len(seq.token_ids)
                    start = seq.num_computed_tokens
                    level_parts.append(level_embed[off + start:off + start + chunk_size])
                    off += seq_len
                sliced_ds.append(torch.cat(level_parts, dim=0))
        return sliced, sliced_ds

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

        for i, (prompt, sp) in enumerate(zip(prompts, sp_list)):
            img = images[i] if i < len(images) else None
            vid = videos[i] if i < len(videos) else None

            if isinstance(prompt, dict) and "token_ids" in prompt:
                pp = prompt
                ids = pp["token_ids"]
                seq = Sequence(ids, max_tokens=sp.max_tokens, ignore_eos=sp.ignore_eos)
                seq.pixel_values = pp.get("pixel_values")
                seq.image_grid_thw = pp.get("image_grid_thw")
                seq.video_pixel_values = pp.get("video_pixel_values")
                seq.video_grid_thw = pp.get("video_grid_thw")
                seq.mrope_positions = pp.get("mrope_positions")
                seq.mrope_position_delta = pp.get("mrope_position_delta", 0)
            elif self.is_qwen_vl and (img is not None or vid is not None):
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
        num_prompts = len(prompts)

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

        def _finish_seq(seq: Sequence) -> None:
            nonlocal _pbar_pending, _pbar_pending_in, _pbar_pending_out
            seq.status = SeqStatus.FINISHED
            seq._cached_embeds = None
            seq._cached_deepstack = None
            bm.deallocate(seq)
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

                    # Launch async: GPU starts working, CPU is free
                    if mr.world_size > 1:
                        mr._write_decode_shm(*decode_data)
                        mr.shm.buf[mr._SHM_FLAG_OFFSET] = 1
                        mr._signal_workers()
                    has_result, _ = mr.run_decode_greedy_fast_async(decode_data)
                    if _PROFILE:
                        _fp_t2 = time.perf_counter()

                    # Wait for first step result (no prior work to overlap yet)
                    if has_result:
                        token_ids = mr._wait_async_tokens(n_dc)
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

                        # --- Pipelined inner decode loop ---
                        # Overlap GPU step N+1 with CPU bookkeeping from step N.
                        use_incr = True
                        prev_token_ids = None
                        prev_decode_seqs = None
                        prev_any_finished = False
                        pending_gpu = False

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

                            # Launch GPU async for this step
                            if mr.world_size > 1:
                                mr._write_decode_shm(*decode_data)
                                mr.shm.buf[mr._SHM_FLAG_OFFSET] = 1
                                mr._signal_workers()
                            has_result, _ = mr.run_decode_greedy_fast_async(
                                decode_data)
                            if _PROFILE:
                                _fp_t2 = time.perf_counter()

                            # Wait for GPU result and process
                            if has_result:
                                token_ids = mr._wait_async_tokens(n_dc)
                                if _PROFILE:
                                    _fp_t3 = time.perf_counter()
                                any_finished = False
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
            while waiting and token_budget > 0:
                seq = waiting[0]
                prompt_len = seq.num_prompt_tokens
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

            if not decode_seqs and not prefill_seqs:
                continue

            # =============================================================
            # EXECUTE: single forward pass
            # =============================================================
            n_pf = len(prefill_seqs)
            n_dc = len(decode_seqs)

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
                    or s._cached_embeds is not None
                    for s in prefill_seqs
                )
                if has_mm:
                    full_embeds, full_ds = self._run_vision_encoder(prefill_seqs)
                    is_chunked = any(
                        cs < len(s.token_ids) - s.num_computed_tokens
                        or s.num_computed_tokens > 0
                        for s, cs in zip(prefill_seqs, prefill_chunk_sizes)
                    )
                    if is_chunked:
                        chunk_embeds, chunk_ds = self._slice_mm_embeds(
                            prefill_seqs, prefill_chunk_sizes,
                            full_embeds, full_ds,
                        )
                        input_ids_t, positions_t = self.model_runner.prepare_mixed_batch(
                            prefill_seqs, prefill_chunk_sizes, [],
                        )
                    else:
                        chunk_embeds, chunk_ds = full_embeds, full_ds
                        input_ids_t, positions_t = self.model_runner.prepare_prefill(prefill_seqs)
                    logits = self.model_runner.run_model(
                        input_ids_t, positions_t, True,
                        inputs_embeds=chunk_embeds,
                        deepstack_embeds=chunk_ds,
                    )
                    reset_context()
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
                    or s._cached_embeds is not None
                    for s in prefill_seqs
                )
                if has_mm:
                    full_embeds, full_ds = self._run_vision_encoder(prefill_seqs)
                    chunk_embeds, chunk_ds = self._slice_mm_embeds(
                        prefill_seqs, prefill_chunk_sizes,
                        full_embeds, full_ds,
                    )
                    input_ids_t, positions_t = self.model_runner.prepare_mixed_batch(
                        prefill_seqs, prefill_chunk_sizes, decode_seqs,
                    )
                    n_prefill_tokens = chunk_embeds.size(0)
                    n_decode_tokens = len(decode_seqs)
                    model = self.model_runner.model
                    decode_ids = input_ids_t[n_prefill_tokens:]
                    decode_embeds = model.get_input_embeddings()(decode_ids)
                    combined_embeds = torch.cat([chunk_embeds, decode_embeds], dim=0)
                    combined_ds = None
                    if chunk_ds is not None:
                        combined_ds = []
                        for level_embed in chunk_ds:
                            pad = torch.zeros(
                                n_decode_tokens, level_embed.size(1),
                                dtype=level_embed.dtype, device=level_embed.device,
                            )
                            combined_ds.append(torch.cat([level_embed, pad], dim=0))
                    logits = self.model_runner.run_model(
                        input_ids_t, positions_t, True,
                        inputs_embeds=combined_embeds,
                        deepstack_embeds=combined_ds,
                    )
                    reset_context()
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
                seq._cached_embeds = None
                seq._cached_deepstack = None
                sample_seqs.append(seq)
                sample_logits.append(logits[i:i+1])
            else:
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
