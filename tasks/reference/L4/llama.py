"""Standalone Llama 3.1 model implementation.

Uses Llama 3.1-style RoPE with frequency scaling, SwiGLU MLP,
and GQA attention. Supports tensor parallelism via shared TP layers.
"""


from __future__ import annotations


# Inlined from infra/context.py
import enum
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn


class CUDAGraphMode(enum.IntEnum):
    """Runtime mode for CUDA graph dispatch (mirrors vLLM CUDAGraphMode)."""
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class AttnBackendConfig:
    """Selects attention backend and associated KV cache parameters.

    Blackwell (sm_100+) uses TRTLLM-gen kernels via FlashInfer (HND layout,
    block_size=16).  Hopper and below use flash_attn (NHD layout,
    block_size=256).  Auto-detection picks the optimal backend for the
    current GPU.
    """
    backend: str = "flash_attn"
    block_size: int = 256
    kv_layout: str = "NHD"

    @classmethod
    def auto_detect(cls) -> "AttnBackendConfig":
        if not torch.cuda.is_available():
            return cls()
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 10:
            try:
                from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: F401
                return cls(backend="trtllm", block_size=16, kv_layout="HND")
            except ImportError:
                pass
        return cls()

    @property
    def use_trtllm(self) -> bool:
        return self.backend == "trtllm"


_ATTN_BACKEND_CONFIG: AttnBackendConfig | None = None


def get_attn_backend_config() -> AttnBackendConfig:
    global _ATTN_BACKEND_CONFIG
    if _ATTN_BACKEND_CONFIG is None:
        _ATTN_BACKEND_CONFIG = AttnBackendConfig.auto_detect()
    return _ATTN_BACKEND_CONFIG


def set_attn_backend_config(config: AttnBackendConfig) -> None:
    global _ATTN_BACKEND_CONFIG
    _ATTN_BACKEND_CONFIG = config


@dataclass
class ChunkedContextMetadata:
    """Metadata for chunked prefill context processing (MLA).

    When a prefill request has prior computed tokens in the KV cache,
    those tokens must be gathered and attended to in chunks to bound
    workspace memory.  Matches vllm's MLACommonPrefillMetadata.ChunkedContextMetadata.
    """
    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens: torch.Tensor
    workspace: torch.Tensor
    token_to_seq: torch.Tensor
    chunk_total_token: list[int]


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    max_context_len: int = 0

    # Chunked prefill: mixed batch with both prefill and decode tokens.
    # Token layout: [prefill_tokens... | decode_tokens...]
    is_mixed: bool = False
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefill_seqs: int = 0

    # Prefill-specific metadata (indexed over prefill seqs only)
    prefill_cu_seqlens_q: torch.Tensor | None = None
    prefill_cu_seqlens_k: torch.Tensor | None = None
    prefill_max_seqlen_q: int = 0
    prefill_max_seqlen_k: int = 0
    prefill_block_tables: torch.Tensor | None = None

    # Decode-specific metadata (indexed over decode seqs only)
    decode_context_lens: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    decode_max_context_len: int = 0

    # Flat indices into concatenated input for extracting one logit per seq
    logit_indices: torch.Tensor | None = None

    # MLA chunked prefill context (for requests with prior computed tokens)
    chunked_context: ChunkedContextMetadata | None = None

    # Per-token request ID mapping (for sparse indexer index conversion)
    req_id_per_token: torch.Tensor | None = None

    # Cross-attention metadata (encoder-decoder models like Whisper)
    # Slot mapping for writing encoder K/V to paged cache
    cross_slot_mapping: torch.Tensor | None = None
    # Prefill: cu_seqlens for decoder Q and encoder K
    cross_cu_seqlens_q: torch.Tensor | None = None
    cross_cu_seqlens_k: torch.Tensor | None = None
    cross_max_seqlen_q: int = 0
    cross_max_seqlen_k: int = 0
    cross_block_tables: torch.Tensor | None = None
    # Decode: context lens = encoder sequence lengths per request
    cross_context_lens: torch.Tensor | None = None
    cross_max_context_len: int = 0

    # --- Compilation / CUDA-graph fields (mirror vLLM ForwardContext) ---
    # Maps layer prefix -> live nn.Module for custom-op runtime lookup.
    no_compile_layers: dict[str, "nn.Module"] = field(default_factory=dict)
    # Runtime mode for CUDAGraphWrapper dispatch.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    # Batch size key used by CUDAGraphWrapper for per-shape graph caching.
    batch_size_for_graph: int = 0

    # --- Mamba / SSM fields (mirror vLLM ForwardContext.attn_metadata
    # for Mamba layers).  ``mamba_state`` owns the global conv/ssm state
    # tensors; ``mamba_metadata`` is a per-batch dataclass (Mamba2Metadata
    # or MambaMetadata) carrying state slot indices and prefill/decode
    # metadata read by every Mamba mixer in its forward pass.
    mamba_state: object = None
    mamba_metadata: object = None


# Global module registry populated once at model init; copied into each
# Context so compiled custom ops can resolve their target modules.
_STATIC_NO_COMPILE_LAYERS: dict[str, "nn.Module"] = {}


def register_no_compile_layers(layers: dict[str, "nn.Module"]) -> None:
    """Register attention/MoE modules for custom-op lookup during compiled
    execution.  Called once after model construction."""
    _STATIC_NO_COMPILE_LAYERS.update(layers)


def get_no_compile_layers() -> dict[str, "nn.Module"]:
    return _STATIC_NO_COMPILE_LAYERS


def auto_register_no_compile_layers(model: "nn.Module") -> None:
    """Walk *model* and register every MoE and Attention sub-module by its
    fully-qualified prefix so custom ops can find them at runtime.

    Recognized types (by class name to avoid circular imports):
      - ``Qwen3MoE``, ``MixtralMoE``, ``GptOssMoE``, ``DeepSeekMoE`` (MoE blocks)
      - ``Attention``, ``MLAAttention``, ``SparseAttnIndexer``       (attention impls)
      - ``Mamba2Mixer``                                              (Mamba2 compile boundary)

    Also sets ``_layer_name`` on each module so it knows its own key.
    ``_use_custom_op`` remains ``False`` until compilation is enabled.
    """
    _TARGET_NAMES = {
        "Qwen3MoE", "MixtralMoE", "GptOssMoE", "DeepSeekMoE",
        "Attention", "MLAAttention", "SparseAttnIndexer",
        "Mamba2Mixer",
    }
    layers: dict[str, "nn.Module"] = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ in _TARGET_NAMES:
            layers[name] = mod
            mod._layer_name = name  # type: ignore[attr-defined]
    register_no_compile_layers(layers)


def enable_custom_ops() -> None:
    """Switch all registered no-compile layers to dispatch through custom ops.
    Called once after torch.compile is applied to the model."""
    for mod in _STATIC_NO_COMPILE_LAYERS.values():
        mod._use_custom_op = True  # type: ignore[attr-defined]


def disable_custom_ops() -> None:
    """Revert to eager dispatch (used for testing/fallback)."""
    for mod in _STATIC_NO_COMPILE_LAYERS.values():
        mod._use_custom_op = False  # type: ignore[attr-defined]


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None,
                max_context_len=0, chunked_context=None,
                req_id_per_token=None):
    global _CONTEXT
    # For pure-decode batches (``is_prefill=False`` with no mixed fields),
    # mirror the generic ``context_lens`` / ``block_tables`` / ``max_context_len``
    # into the decode-specific fields so that DSA indexer and other
    # decode-specialised paths (which consult ``decode_context_lens`` /
    # ``decode_block_tables`` — matching vLLM's FlashInfer metadata) can
    # find them.  Without this, ``SparseAttnIndexer._decode_topk`` would
    # early-return all -1 indices and attention would degenerate.
    dc_cl = context_lens if not is_prefill else None
    dc_bt = block_tables if not is_prefill else None
    dc_max = max_context_len if not is_prefill else 0
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables, max_context_len,
                       chunked_context=chunked_context,
                       req_id_per_token=req_id_per_token,
                       no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
                       decode_context_lens=dc_cl,
                       decode_block_tables=dc_bt,
                       decode_max_context_len=dc_max)


def set_mixed_context(
    slot_mapping,
    num_prefill_tokens, num_decode_tokens, num_prefill_seqs,
    prefill_cu_seqlens_q, prefill_cu_seqlens_k,
    prefill_max_seqlen_q, prefill_max_seqlen_k,
    prefill_block_tables,
    decode_context_lens, decode_block_tables, decode_max_context_len,
    logit_indices,
    chunked_context=None,
    req_id_per_token=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=True, is_mixed=True,
        slot_mapping=slot_mapping,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefill_seqs=num_prefill_seqs,
        prefill_cu_seqlens_q=prefill_cu_seqlens_q,
        prefill_cu_seqlens_k=prefill_cu_seqlens_k,
        prefill_max_seqlen_q=prefill_max_seqlen_q,
        prefill_max_seqlen_k=prefill_max_seqlen_k,
        prefill_block_tables=prefill_block_tables,
        decode_context_lens=decode_context_lens,
        decode_block_tables=decode_block_tables,
        decode_max_context_len=decode_max_context_len,
        logit_indices=logit_indices,
        chunked_context=chunked_context,
        req_id_per_token=req_id_per_token,
        no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
    )


def set_mamba_context(
    is_prefill: bool,
    mamba_state,
    mamba_metadata,
):
    """Install per-batch Mamba state + metadata in the global Context.

    Used by ``ModelRunner.run_mamba`` -- mirrors how the attention path
    uses ``set_context`` / ``set_mixed_context``.  Mamba mixers read
    ``ctx.mamba_state`` and ``ctx.mamba_metadata`` in their forward.
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        mamba_state=mamba_state,
        mamba_metadata=mamba_metadata,
        no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context(no_compile_layers=_STATIC_NO_COMPILE_LAYERS)


@contextmanager
def set_forward_context(
    is_prefill: bool = False,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_size_for_graph: int = 0,
    **ctx_kwargs,
):
    """Context manager that sets both KV-cache metadata and compile/graph
    fields for the duration of a forward pass.

    ``no_compile_layers`` is always populated from the global registry so
    custom ops can resolve modules without the caller threading it through.
    """
    global _CONTEXT
    prev = _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        no_compile_layers=_STATIC_NO_COMPILE_LAYERS,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_size_for_graph=batch_size_for_graph,
        **ctx_kwargs,
    )
    try:
        yield _CONTEXT
    finally:
        _CONTEXT = prev


# Inlined from infra/tp.py
import torch.distributed as dist


def _tp_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def _tp_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# Inlined from tasks/reference/L1/linear.py
import torch.nn as nn
import torch.nn.functional as F


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


# Inlined from tasks/reference/L1/embedding.py
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=padding_idx)

    def forward(self, input_ids):
        return self.emb(input_ids)


# Inlined from tasks/reference/L1/allreduce.py
from contextlib import nullcontext
from typing import Optional

from torch.distributed import ProcessGroup


_CUSTOM_AR: Optional["CustomAllreduce"] = None


def set_custom_ar(ar):
    global _CUSTOM_AR
    _CUSTOM_AR = ar


def get_custom_ar():
    return _CUSTOM_AR


class AllReduce(nn.Module):
    def forward(self, tensor):
        dist.all_reduce(tensor)
        return tensor


class CustomAllreduce:
    """Compatibility shim for callers expecting the baseline custom AR API."""

    disabled = True

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = 8192 * 1024,
    ) -> None:
        del group, device, max_size

    def capture(self):
        return nullcontext()

    def custom_all_reduce(self, input: torch.Tensor) -> None:
        del input
        return None

    def close(self) -> None:
        pass

__all__ = ["AllReduce", "CustomAllreduce", "get_custom_ar", "set_custom_ar"]


# Inlined from tasks/reference/L2/parallel_embedding.py
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__()
        tp, rank = _tp_size(), _tp_rank()
        assert num_embeddings % tp == 0
        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.padding_size = padding_size
        self.embedding_dim = embedding_dim
        self.per_partition = num_embeddings // tp
        self.vocab_start = self.per_partition * rank
        self.vocab_end = self.vocab_start + self.per_partition
        self.tp_size = tp
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.embedding_op = Embedding(self.per_partition, embedding_dim)
        self.embedding_op.emb.weight.weight_loader = self._weight_loader
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start) & (x < self.vocab_end)
            x = mask * (x - self.vocab_start)
        y = self.embedding_op(x)
        if self.tp_size > 1:
            y = mask.unsqueeze(-1) * y
            y = self.allreduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 bias: bool = False,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__(num_embeddings, embedding_dim,
                         params_dtype=params_dtype,
                         org_num_embeddings=org_num_embeddings,
                         padding_size=padding_size)
        self.linear_op = Matmul()

    def project(self, x):
        """Linear projection only (no gather). Used inside CUDA graph."""
        ctx = get_context()
        if ctx.is_mixed:
            x = x[ctx.logit_indices].contiguous()
        elif ctx.is_prefill:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return self.linear_op(x, self.embedding_op.emb.weight)

    def gather_logits(self, logits):
        """Gather partial logits from all ranks. Used outside CUDA graph."""
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if _tp_rank() == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if _tp_rank() == 0 else logits
        return logits

    def gather_greedy(self, logits):
        """Fast path for greedy: local argmax + small allgather.

        Instead of gathering full vocab logits (~31MB/rank), gather only
        the (max_val, max_idx) per sequence (~2KB/rank).
        Returns token IDs directly on rank 0, None on other ranks.
        """
        if self.tp_size <= 1:
            return None

        rank = _tp_rank()
        local_max_vals, local_max_idxs = logits.max(dim=-1)
        local_max_idxs = local_max_idxs + self.vocab_start

        info = torch.stack([local_max_vals, local_max_idxs.float()], dim=-1)
        gathered = [torch.empty_like(info) for _ in range(self.tp_size)]
        dist.all_gather(gathered, info)
        if rank == 0:
            all_info = torch.stack(gathered, dim=0)
            all_vals = all_info[:, :, 0]
            all_idxs = all_info[:, :, 1].long()
            best_rank = all_vals.argmax(dim=0)
            bs = logits.size(0)
            token_ids = all_idxs[best_rank, torch.arange(bs, device=logits.device)]
            return token_ids
        return None

    def forward(self, x):
        logits = self.project(x)
        return self.gather_logits(logits)


# Inlined from tasks/reference/L1/rms_norm.py
import os

from torch.utils.cpp_extension import load_inline


_CPP_SRC = r"""
#include <torch/extension.h>

void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps);
void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "RMSNorm");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add RMSNorm");
}
"""


_CUDA_SRC = r"""
// Standalone RMSNorm and fused-add-RMSNorm CUDA kernels for fastkernels.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <numeric>
#include <type_traits>
#include <torch/all.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define DISPATCH_CASE_FLOAT_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))

namespace {

struct CubAddOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

template <typename torch_type>
struct TypeConvert {
  static constexpr bool exists = false;
};

template <>
struct TypeConvert<float> {
  static constexpr bool exists = true;
  using device_type = float;
  using packed_type = float2;
  __device__ static __forceinline__ float convert(device_type x) { return x; }
};

template <>
struct TypeConvert<c10::Half> {
  static constexpr bool exists = true;
  using device_type = __half;
  using packed_type = __half2;
  __device__ static __forceinline__ float convert(device_type x) {
    return __half2float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_type x) {
    return __half22float2(x);
  }
  __device__ static __forceinline__ device_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static __forceinline__ packed_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

template <>
struct TypeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using device_type = __nv_bfloat16;
  using packed_type = __nv_bfloat162;
  __device__ static __forceinline__ float convert(device_type x) {
    return __bfloat162float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static __forceinline__ device_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static __forceinline__ packed_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};

template <typename scalar_t, int width>
struct alignas(16) F16Vec {
  static_assert(width > 0 && (width & (width - 1)) == 0);
  using Converter = TypeConvert<scalar_t>;
  using T1 = typename Converter::device_type;
  using T2 = typename Converter::packed_type;
  T1 data[width];

  __device__ F16Vec& operator+=(const F16Vec& other) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      if constexpr (std::is_same_v<T2, float2>) {
        data[i] += other.data[i];
        data[i + 1] += other.data[i + 1];
      } else {
        T2 temp{data[i], data[i + 1]};
        temp += T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    }
    return *this;
  }

  __device__ F16Vec& operator*=(const F16Vec& other) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      if constexpr (std::is_same_v<T2, float2>) {
        data[i] *= other.data[i];
        data[i + 1] *= other.data[i + 1];
      } else {
        T2 temp{data[i], data[i + 1]};
        temp *= T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    }
    return *this;
  }

  __device__ F16Vec& operator*=(const float scale) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
      temp_f.x *= scale;
      temp_f.y *= scale;
      T2 temp = Converter::convert(temp_f);
      data[i] = temp.x;
      data[i + 1] = temp.y;
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 z = Converter::convert(T2{data[i], data[i + 1]});
      result += z.x * z.x + z.y * z.y;
    }
    return result;
  }
};

template <int VEC_SIZE, typename scalar_t, typename VecOp, typename ScalarOp>
__device__ inline void vectorize_read_with_alignment(
    const scalar_t* input,
    int len,
    int tid,
    int stride,
    VecOp&& vec_op,
    ScalarOp&& scalar_op) {
  constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
  uintptr_t addr = reinterpret_cast<uintptr_t>(input);
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;
    auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input);
    for (int i = tid; i < num_vec; i += stride) {
      vec_op(v_in[i]);
    }
    return;
  }
  int misalignment_offset = addr & (WIDTH - 1);
  int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = (alignment_bytes & (WIDTH - 1)) / sizeof(scalar_t);
  prefix_elems = min(prefix_elems, len);
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(input[i]);
  }
  input += prefix_elems;
  len -= prefix_elems;
  int num_vec = len / VEC_SIZE;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input);
  for (int i = tid; i < num_vec; i += stride) {
    vec_op(v_in[i]);
  }
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(input[i]);
  }
}

}  // namespace

template <typename scalar_t>
__global__ void rmsnorm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float eps,
    const int hidden_size) {
  const int token = blockIdx.x;
  const scalar_t* x = input + token * hidden_size;
  scalar_t* o = out + token * hidden_size;

  float sum_sq = 0.0f;
  auto vec_op = [&sum_sq](const vec_n_t<scalar_t, 8>& vec) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = static_cast<float>(vec.val[i]);
      sum_sq += v * v;
    }
  };
  auto scalar_op = [&sum_sq](const scalar_t& val) {
    float v = static_cast<float>(val);
    sum_sq += v * v;
  };
  vectorize_read_with_alignment<8>(x, hidden_size, threadIdx.x, blockDim.x,
                                   vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  sum_sq = BlockReduce(reduce_store).Reduce(sum_sq, CubAddOp{}, blockDim.x);

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, 8>*>(x);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, 8>*>(weight);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, 8>*>(o);
  for (int i = threadIdx.x; i < hidden_size / 8; i += blockDim.x) {
    vec_n_t<scalar_t, 8> dst;
    vec_n_t<scalar_t, 8> src1 = v_in[i];
    vec_n_t<scalar_t, 8> src2 = v_w[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float v = static_cast<float>(src1.val[j]);
      dst.val[j] = static_cast<scalar_t>(v * s_rms_inv) * src2.val[j];
    }
    v_out[i] = dst;
  }
}

void rmsnorm(
    torch::Tensor& output,
    torch::Tensor& input,
    torch::Tensor& weight,
    double eps) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(input.scalar_type(), "rmsnorm_kernel", [&] {
    rmsnorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        hidden_size);
  });
}

template <typename scalar_t>
__global__ void fused_add_rmsnorm_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float eps,
    const int hidden_size) {
  const int token = blockIdx.x;
  scalar_t* x = input + token * hidden_size;
  scalar_t* r = residual + token * hidden_size;

  // Step 1: residual += input; then compute rms on residual
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float ri = static_cast<float>(r[i]) + static_cast<float>(x[i]);
    r[i] = static_cast<scalar_t>(ri);
    sum_sq += ri * ri;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  sum_sq = BlockReduce(reduce_store).Reduce(sum_sq, CubAddOp{}, blockDim.x);

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  // Step 2: input = rmsnorm(residual) * weight
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float ri = static_cast<float>(r[i]);
    x[i] = static_cast<scalar_t>(ri * s_rms_inv) * weight[i];
  }
}

template <typename scalar_t, int width>
__global__ std::enable_if_t<TypeConvert<scalar_t>::exists>
fused_add_rmsnorm_vec_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float eps,
    const int hidden_size,
    const int64_t input_stride) {
  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  float sum_sq = 0.0f;

  auto* __restrict__ input_v = reinterpret_cast<F16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v = reinterpret_cast<F16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v = reinterpret_cast<const F16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    F16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    sum_sq += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  sum_sq = BlockReduce(reduce_store).Reduce(sum_sq, CubAddOp{}, blockDim.x);

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    F16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_rms_inv;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

void fused_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes = vector_width * 2;
  auto inp_ptr = reinterpret_cast<uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input.stride(-2) % vector_width == 0;
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
      (input.scalar_type() == at::ScalarType::Half ||
       input.scalar_type() == at::ScalarType::BFloat16)) {
    AT_DISPATCH_SWITCH(
        input.scalar_type(), "fused_add_rmsnorm_vec_kernel",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
          fused_add_rmsnorm_vec_kernel<scalar_t, vector_width><<<grid, block, 0, stream>>>(
              input.data_ptr<scalar_t>(),
              residual.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              static_cast<float>(eps),
              hidden_size,
              input.stride(-2));
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
          fused_add_rmsnorm_vec_kernel<scalar_t, vector_width><<<grid, block, 0, stream>>>(
              input.data_ptr<scalar_t>(),
              residual.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              static_cast<float>(eps),
              hidden_size,
              input.stride(-2));
        }));
    return;
  }

  DISPATCH_FLOAT_TYPES(input.scalar_type(), "fused_add_rmsnorm_kernel", [&] {
    fused_add_rmsnorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        hidden_size);
  });
}
"""


_INLINE_EXT = None


def _load_inline_ext():
    global _INLINE_EXT
    if _INLINE_EXT is None:
        extra_cuda_cflags = [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ]
        build_directory = os.path.join(
            os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions"),
            "fastkernels_reference_rmsnorm_inline",
        )
        os.makedirs(build_directory, exist_ok=True)
        _INLINE_EXT = load_inline(
            name="fastkernels_reference_rmsnorm_inline",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=build_directory,
            verbose=bool(int(os.environ.get("FASTKERNELS_VERBOSE_EXT", "0"))),
        )
    return _INLINE_EXT


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    @staticmethod
    def forward_native(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        hidden_size: int,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        if residual is not None:
            residual_tensor = residual
            residual = (x + residual).to(orig_dtype)
            residual_tensor.copy_(residual)
            x_float = residual.float()
        else:
            x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        out = (x_float * torch.rsqrt(variance + eps)).to(orig_dtype)
        if weight is not None:
            out = out * weight
        if residual is None:
            return out
        x.copy_(out)
        return out, residual

    @staticmethod
    def forward_cuda(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if weight is not None and x.is_cuda:
            ext = _load_inline_ext()
            if residual is None:
                out = torch.empty_like(x)
                ext.rmsnorm(out, x, weight, eps)
                return out
            ext.fused_add_rmsnorm(x, residual, weight, eps)
            return x, residual
        if weight is None and residual is None:
            return F.rms_norm(x, (x.size(-1),), eps=eps)
        return RMSNorm.forward_native(x, weight, eps, x.size(-1), residual)

    def forward(self, x, residual=None):
        return self.forward_cuda(
            x,
            self.weight if self.elementwise_affine else None,
            self.eps,
            residual,
        )


# Inlined from tasks/reference/L1/rotary_emb.py
import math


def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    smooth = (
        (original_max_position_embeddings / wl - low_freq_factor)
        / (high_freq_factor - low_freq_factor)
        if low_freq_factor != high_freq_factor
        else torch.zeros_like(inv_freq)
    )
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim)
        )
        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        self.register_buffer(
            "cos_sin_cache", torch.cat((freqs.cos(), freqs.sin()), dim=-1).float(),
            persistent=False,
        )

    @staticmethod
    def forward_native(positions, query, key, head_dim, cos_sin_cache):
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim].unsqueeze(1)
        sin = cos_sin[..., embed_dim:].unsqueeze(1)
        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)
        q1, q2 = q[..., :embed_dim], q[..., embed_dim:]
        k1, k2 = k[..., :embed_dim], k[..., embed_dim:]
        query_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        key_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return query_out.view(q_shape), key_out.view(k_shape)

    def forward_cuda(self, positions, query, key):
        cache = self.cos_sin_cache.to(query.dtype)
        query_out, key_out = self.forward_native(positions, query, key, self.head_dim, cache)
        query.copy_(query_out)
        key.copy_(key_out)
        return query, key

    def forward(self, positions, query, key):
        return self.forward_cuda(positions, query, key)


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


# Inlined from tasks/reference/L1/flash_attn_decode.py
class FlashAttnDecode(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        block_table = kwargs.get("block_table", None)
        softmax_scale = kwargs.get("softmax_scale", self.head_dim ** -0.5)
        window_size = kwargs.get("window_size", (-1, -1))
        window_size = (-1, -1) if window_size is None else tuple(window_size)
        s_aux = kwargs.get("s_aux", None)
        softcap = kwargs.get("softcap", 0.0)
        if cache_seqlens is None:
            cache_seqlens = torch.full((q.shape[0],), k_cache.shape[0], device=q.device, dtype=torch.int32)

        outs = []
        for i in range(q.shape[0]):
            seq_len = int(cache_seqlens[i].item())
            k = gather_paged_cache(k_cache, block_table, i, seq_len)
            v = gather_paged_cache(v_cache, block_table, i, seq_len)
            out = dense_attention(
                q[i:i + 1].unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                softmax_scale=softmax_scale, causal=True,
                window_size=window_size, s_aux=s_aux, softcap=softcap,
            ).squeeze(0).squeeze(0)
            outs.append(out)
        return torch.stack(outs, dim=0)


# Inlined from tasks/reference/L1/flash_attn_prefill.py
class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        del max_seqlen_q, max_seqlen_k
        block_table = kwargs.get("block_table")
        window_size = kwargs.get("window_size", (-1, -1))
        window_size = (-1, -1) if window_size is None else tuple(window_size)
        if block_table is not None and k.ndim == 4:
            k_parts = []
            v_parts = []
            cu_k = [0]
            for i in range(cu_seqlens_k.numel() - 1):
                seq_len = int((cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item())
                k_seq = gather_paged_cache(k, block_table, i, seq_len)
                v_seq = gather_paged_cache(v, block_table, i, seq_len)
                k_parts.append(k_seq)
                v_parts.append(v_seq)
                cu_k.append(cu_k[-1] + k_seq.shape[0])
            k = torch.cat(k_parts, dim=0) if k_parts else k.new_empty((0, self.num_kv_heads, self.head_dim))
            v = torch.cat(v_parts, dim=0) if v_parts else v.new_empty((0, self.num_kv_heads, self.head_dim))
            cu_seqlens_k = torch.tensor(cu_k, device=cu_seqlens_k.device, dtype=cu_seqlens_k.dtype)
        return varlen_attention(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=kwargs.get("softmax_scale", self.sm_scale),
            causal=kwargs.get("causal", True),
            window_size=window_size,
            s_aux=kwargs.get("s_aux", None),
            softcap=kwargs.get("softcap", 0.0),
        )


# Inlined from tasks/reference/L1/flashinfer_decode.py
class TRTLLMDecode(nn.Module):
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        workspace: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self._workspace = workspace

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, **kwargs):
        del causal, max_seq_len, kwargs
        if cache_seqlens is None:
            cache_seqlens = torch.full((q.shape[0],), k_cache.shape[2], device=q.device, dtype=torch.int32)
        scale = softmax_scale if softmax_scale is not None else self.sm_scale
        outs = []
        for i in range(q.shape[0]):
            seq_len = int(cache_seqlens[i].item())
            k = gather_paged_cache(k_cache, block_table, i, seq_len, hnd=True)
            v = gather_paged_cache(v_cache, block_table, i, seq_len, hnd=True)
            out = dense_attention(
                q[i:i + 1].unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                softmax_scale=scale, causal=False,
            ).squeeze(0).squeeze(0)
            outs.append(out)
        return torch.stack(outs, dim=0)


# Inlined from tasks/reference/L1/flashinfer_prefill.py
class TRTLLMPrefill(nn.Module):
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        workspace: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self._workspace = workspace

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale=None,
                causal=True, block_table=None, **kwargs):
        del max_seqlen_q, max_seqlen_k, kwargs
        if block_table is not None and k.ndim == 4:
            k_parts = []
            v_parts = []
            cu_k = [0]
            for i in range(cu_seqlens_k.numel() - 1):
                seq_len = int((cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item())
                k_seq = gather_paged_cache(k, block_table, i, seq_len, hnd=True)
                v_seq = gather_paged_cache(v, block_table, i, seq_len, hnd=True)
                k_parts.append(k_seq)
                v_parts.append(v_seq)
                cu_k.append(cu_k[-1] + k_seq.shape[0])
            k = torch.cat(k_parts, dim=0) if k_parts else k.new_empty((0, self.num_kv_heads, self.head_dim))
            v = torch.cat(v_parts, dim=0) if v_parts else v.new_empty((0, self.num_kv_heads, self.head_dim))
            cu_seqlens_k = torch.tensor(cu_k, device=cu_seqlens_k.device, dtype=cu_seqlens_k.dtype)
        return varlen_attention(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            causal=causal,
        )


# Inlined from tasks/reference/L1/store_kvcache.py
class StoreKVCache(nn.Module):
    """NHD layout store: [num_blocks, block_size, num_kv_heads, head_dim]."""

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        flat_k = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
        flat_v = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])
        valid = slot_mapping >= 0
        slots = slot_mapping[valid].long()
        flat_k.index_copy_(0, slots, key[valid])
        flat_v.index_copy_(0, slots, value[valid])


class StoreKVCacheHND(nn.Module):
    """HND layout store: [num_blocks, num_kv_heads, block_size, head_dim]."""

    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        valid = slot_mapping >= 0
        slots = slot_mapping[valid].long()
        block_idx = slots // self.page_size
        slot_in_block = slots % self.page_size
        k_cache[block_idx, :, slot_in_block, :] = key[valid]
        v_cache[block_idx, :, slot_in_block, :] = value[valid]


# Inlined from tasks/reference/L1/fp8_linear.py
_GROUP_SIZE = 128


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _expand_weight_scale(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight_fp8.shape[-2], weight_fp8.shape[-1]
    row_blocks = _ceil_div(rows, _GROUP_SIZE)
    col_blocks = _ceil_div(cols, _GROUP_SIZE)
    scale_f = scale.float()
    if scale_f.shape[-2:] == (row_blocks, col_blocks):
        expanded = scale_f.repeat_interleave(_GROUP_SIZE, dim=-2)
        expanded = expanded.repeat_interleave(_GROUP_SIZE, dim=-1)
        return expanded[..., :rows, :cols]
    if scale_f.shape[-1] == col_blocks:
        expanded = scale_f.repeat_interleave(_GROUP_SIZE, dim=-1)
        return expanded[..., :cols].unsqueeze(-2).expand_as(weight_fp8.float())
    return scale_f.expand_as(weight_fp8.float())


def _quantize_fp8_per_token_group(
    source: torch.Tensor,
    out_fp8: torch.Tensor,
    out_scale: torch.Tensor,
    *,
    use_ue8m0: bool = True,
    eps: float = 1e-10,
) -> None:
    info = torch.finfo(torch.float8_e4m3fn)
    flat = source.reshape(-1, source.shape[-1]).float()
    groups = _ceil_div(flat.shape[-1], _GROUP_SIZE)
    padded_cols = groups * _GROUP_SIZE
    if padded_cols != flat.shape[-1]:
        padded = flat.new_zeros(flat.shape[0], padded_cols)
        padded[:, :flat.shape[-1]] = flat
    else:
        padded = flat
    grouped = padded.view(flat.shape[0], groups, _GROUP_SIZE)
    scale = grouped.abs().amax(dim=-1).clamp_min(eps) / info.max
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    expanded = scale.repeat_interleave(_GROUP_SIZE, dim=-1)[:, :flat.shape[-1]]
    out_fp8.copy_(torch.clamp(flat / expanded, info.min, info.max).to(out_fp8.dtype).view_as(out_fp8))
    out_scale.copy_(scale.view_as(out_scale))


class _Fp8PrefillBufs:
    def __init__(self):
        self.input_fp8 = None
        self.input_scale = None
        self.output = None


class PerTokenGroupQuantFp8(nn.Module):
    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        _quantize_fp8_per_token_group(x, out_fp8, out_scale)


class Fp8Linear(nn.Module):
    BLOCK_SIZE = _GROUP_SIZE
    _FLASHINFER_M_THRESHOLD = 32

    def __init__(self):
        super().__init__()
        self._a_buf = None
        self._s_buf = None
        self._o_buf = None
        self._pf = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = torch.empty(max_tokens, math.ceil(K / _GROUP_SIZE), dtype=torch.float32, device=device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        n, k = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, k)
        weight = weight_fp8.float() * _expand_weight_scale(weight_fp8, weight_scale_inv)
        output = F.linear(input_2d.float(), weight.float(), bias.float() if bias is not None else None)
        return output.to(input_bf16.dtype).view(*input_bf16.shape[:-1], n)


def postprocess_fp8_weights(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return weight_fp8, scale_inv


# Inlined from tasks/reference/L2/parallel_linear.py
def _get_fp8_linear_cls():
    return Fp8Linear

_FP8_BLOCK = 128


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


class ColumnParallelLinear(nn.Module):
    """Splits output dim across TP ranks."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(self.output_size_per_partition, input_size,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(self.output_size_per_partition, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        rows_per_shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * rows_per_shard, rows_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """gate_proj + up_proj merged into one linear, sharded across TP."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 quant_config: dict | None = None, disable_tp: bool = False):
        super().__init__()
        tp = _tp_size()
        self.disable_tp = disable_tp
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        if not disable_tp:
            assert all(s % tp == 0 for s in output_sizes)
        self.use_fp8 = quant_config is not None

        effective_tp = 1 if disable_tp else tp
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(total // effective_tp, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(total // effective_tp, input_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(total // effective_tp, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(total // tp))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: int | None = None):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id is None:
            # Fused weight: ``loaded_weight`` is the full ``[sum(output_sizes), in]``
            # tensor.  Recurse per-shard so each output block is sharded across
            # TP ranks independently (mirrors vLLM's ``MergedColumnParallelLinear``
            # weight loader when called without an explicit shard id).
            offset = 0
            for sid, sz in enumerate(self.output_sizes):
                self._weight_loader(
                    param, loaded_weight.narrow(0, offset, sz), sid,
                )
                offset += sz
            return
        effective_tp = 1 if self.disable_tp else tp
        shard_offset = sum(self.output_sizes[:shard_id]) // effective_tp
        shard_size = self.output_sizes[shard_id] // effective_tp
        dst = param.data.narrow(0, shard_offset, shard_size)
        if self.disable_tp:
            dst.copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: int):
        tp, rank = _tp_size(), _tp_rank()
        effective_tp = 1 if self.disable_tp else tp
        shard_size_out = self.output_sizes[shard_id] // effective_tp
        scale_rows = math.ceil(shard_size_out / _FP8_BLOCK)
        shard_offset_out = sum(self.output_sizes[:shard_id]) // effective_tp
        scale_offset = math.ceil(shard_offset_out / _FP8_BLOCK)
        if self.disable_tp:
            param.data.narrow(0, scale_offset, scale_rows).copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    """Q, K, V projections merged and sharded across TP."""

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        # Replicate KV heads when not evenly divisible by TP
        if total_num_kv_heads % tp == 0:
            self.num_kv_heads = total_num_kv_heads // tp
            self._replicate_kv = False
        else:
            self.num_kv_heads = total_num_kv_heads
            self._replicate_kv = True
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, hidden_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            src = loaded_weight.chunk(tp, 0)[rank]
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        dst = param.data.narrow(0, shard_offset, shard_size)
        dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        scale_rows = math.ceil(shard_size / _FP8_BLOCK)
        scale_offset = math.ceil(shard_offset / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class ReplicatedLinear(nn.Module):
    """Full weight replicated on every TP rank (no sharding, no all-reduce)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.weight_scale_inv.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Splits input dim across TP ranks, all-reduces output."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, reduce_results: bool = True):
        super().__init__()
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.reduce_results = reduce_results
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, self.input_size_per_partition,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, self.input_size_per_partition),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        cols_per_shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * cols_per_shard, cols_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            y = self.linear_op(x, self.weight, self.weight_scale_inv,
                               self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.reduce_results and self.tp_size > 1:
            y = self.allreduce(y)
        return y


# Inlined from tasks/reference/L2/attention_impl.py
import numpy as np


def _chunked_prefill_remap(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]:
    device = cu_seqlens_q.device
    cu_q_np = cu_seqlens_q.cpu().numpy()
    cu_k_np = cu_seqlens_k.cpu().numpy()
    q_seqlens = cu_q_np[1:] - cu_q_np[:-1]
    k_seqlens = cu_k_np[1:] - cu_k_np[:-1]
    q_tokens_in_first_block = np.minimum(
        attention_chunk_size - ((k_seqlens - q_seqlens) % attention_chunk_size),
        q_seqlens,
    ).astype(np.int32)
    tokens_in_last_block = (
        attention_chunk_size + (k_seqlens % -attention_chunk_size)
    ).astype(np.int32)
    local_blocks = (
        1 + np.ceil(
            np.maximum(q_seqlens - q_tokens_in_first_block, 0) / attention_chunk_size
        ).astype(np.int32)
    )
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = int(cu_num_blocks[-1])
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    seqlens_q_local = np.repeat(
        q_seqlens - q_tokens_in_first_block, local_blocks,
    ).astype(np.int32)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attention_chunk_size * (arange - 1),
        attention_chunk_size,
    )[arange > 0]
    cu_q_out = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_q_out[1:])
    cu_q_out[0] = 0
    seqlens_k_local = np.full(virtual_batches, attention_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    cu_k_out = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_k_local, out=cu_k_out[1:])
    cu_k_out[0] = 0
    block_tables_out = None
    if block_tables is not None and block_size > 0:
        pages_per_chunk = attention_chunk_size // block_size
        k_seqstarts_absolute = np.repeat(k_seqlens, local_blocks) - (
            rarange * attention_chunk_size
            + np.repeat(tokens_in_last_block, local_blocks)
        )
        block_starts = k_seqstarts_absolute // block_size
        block_indices = (
            block_starts[:, None]
            + np.arange(pages_per_chunk, dtype=np.int32)
        ).reshape(-1).clip(max=block_tables.shape[1] - 1)
        batch_indices = np.repeat(
            np.arange(len(q_seqlens), dtype=np.int32),
            local_blocks * pages_per_chunk,
        )
        block_tables_out = block_tables[
            torch.from_numpy(batch_indices),
            torch.from_numpy(block_indices),
        ].view(virtual_batches, -1)
    return (
        torch.from_numpy(cu_q_out).to(device=device),
        torch.from_numpy(cu_k_out).to(device=device),
        int(seqlens_q_local.max()) if virtual_batches > 0 else 0,
        int(seqlens_k_local.max()) if virtual_batches > 0 else 0,
        block_tables_out,
    )


def _chunked_decode_remap(
    cache_seqlens: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    local_seqlens = torch.clamp(cache_seqlens, max=attention_chunk_size)
    max_context_len = int(local_seqlens.max().item()) if local_seqlens.numel() > 0 else 0
    if block_tables is not None and block_size > 0:
        pages_per_chunk = attention_chunk_size // block_size
        chunk_start_page = (cache_seqlens - local_seqlens) // block_size
        offsets = torch.arange(pages_per_chunk, device=block_tables.device)
        page_indices = (chunk_start_page.unsqueeze(1) + offsets).clamp(
            max=block_tables.shape[1] - 1,
        )
        block_tables = torch.gather(block_tables, 1, page_indices)
    return local_seqlens, block_tables, max_context_len


class Attention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 sliding_window: int | None = None,
                 sinks: torch.nn.Parameter | None = None,
                 attention_chunk_size: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.sinks = sinks
        self.attention_chunk_size = attention_chunk_size
        self.k_cache = self.v_cache = torch.tensor([])
        attn_cfg = get_attn_backend_config()
        self._use_trtllm = attn_cfg.use_trtllm
        self._block_size = attn_cfg.block_size
        self._fa3_sinks = sinks
        self._fa3_window_size = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )
        if self._use_trtllm:
            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            self.prefill_op = TRTLLMPrefill(self.num_heads, self.num_kv_heads, head_size)
            self.decode_op = TRTLLMDecode(self.num_heads, self.num_kv_heads, head_size)
        else:
            self.store_kvcache = StoreKVCache()
            self.prefill_op = FlashAttnPrefill(self.num_heads, self.num_kv_heads, head_size)
            self.decode_op = FlashAttnDecode(self.num_heads, self.num_kv_heads, head_size)
        self._use_custom_op = False
        self._layer_name = ""

    def set_trtllm_workspace(self, workspace: torch.Tensor):
        if self._use_trtllm:
            self.decode_op._workspace = workspace
            self.prefill_op._workspace = workspace

    def forward_impl(self, query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        n = query.shape[0]
        q = query.view(n, self.num_heads, self.head_size)
        k = key.view(n, self.num_kv_heads, self.head_size)
        v = value.view(n, self.num_kv_heads, self.head_size)
        if self.k_cache.numel() and self.v_cache.numel():
            self.store_kvcache(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)
        if ctx.is_mixed:
            out = self._forward_mixed(q, self.k_cache, self.v_cache, ctx)
        else:
            out = self._forward_pure(q, k, v, self.k_cache, self.v_cache, ctx)
        return out.reshape(n, self.num_heads * self.head_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(query, key, value)

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        if ctx.is_prefill:
            cu_q, cu_k = ctx.cu_seqlens_q, ctx.cu_seqlens_k
            msq, msk = ctx.max_seqlen_q, ctx.max_seqlen_k
            bt = ctx.block_tables
            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )
            return self.prefill_op(
                q, k_cache if bt is not None else k, v_cache if bt is not None else v,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True, block_table=bt, **fa_extra,
            )
        cache_seqlens, bt, max_ctx = ctx.context_lens, ctx.block_tables, ctx.max_context_len
        if self.attention_chunk_size is not None:
            cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                cache_seqlens, bt, self.attention_chunk_size, self._block_size,
            )
        return self.decode_op(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens, block_table=bt,
            softmax_scale=self.scale, causal=True, max_seq_len=max_ctx, **fa_extra,
        )

    def _forward_mixed(self, q, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)
        if np_ > 0:
            out[:np_] = self.prefill_op(
                q[:np_], k_cache, v_cache,
                cu_seqlens_q=ctx.prefill_cu_seqlens_q,
                cu_seqlens_k=ctx.prefill_cu_seqlens_k,
                max_seqlen_q=ctx.prefill_max_seqlen_q,
                max_seqlen_k=ctx.prefill_max_seqlen_k,
                softmax_scale=self.scale, causal=True,
                block_table=ctx.prefill_block_tables, **fa_extra,
            )
        if nd > 0:
            out[np_:] = self.decode_op(
                q[np_:], k_cache, v_cache,
                cache_seqlens=ctx.decode_context_lens,
                block_table=ctx.decode_block_tables,
                softmax_scale=self.scale, causal=True,
                max_seq_len=ctx.decode_max_context_len, **fa_extra,
            )
        return out


# Inlined from tasks/reference/L2/attention.py
class LlamaAttention(nn.Module):
    """Model-level attention: qkv_proj -> [qk_norm] -> [rope] -> Attention -> o_proj."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 rotary_emb: nn.Module | None = None,
                 bias: bool = False,              # Qwen2 / GPT-OSS
                 qk_norm: bool = False,           # Qwen3
                 rms_norm_eps: float = 1e-6,
                 nope: bool = False,              # Llama 4
                 use_weightless_qk_norm: bool = False,   # Llama 4
                 attn_temperature_tuning: bool = False,  # Llama 4
                 floor_scale: float = 8192.0,            # Llama 4
                 attn_scale: float = 0.1,                # Llama 4
                 quant_config: dict | None = None,
                 attention_chunk_size: int | None = None,
                 o_proj_bias: bool = False,              # GPT-OSS
                 use_sinks: bool = False,                # GPT-OSS
                 sliding_window: int | None = None,      # GPT-OSS
                 layer_idx: int = 0):                     # GPT-OSS
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        if num_key_value_heads >= tp:
            self.num_kv_heads = num_key_value_heads // tp
        else:
            self.num_kv_heads = 1
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.nope = nope
        self.attn_temperature_tuning = attn_temperature_tuning and nope
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            bias=o_proj_bias,
            quant_config=quant_config,
        )

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None  # Qwen3
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None  # Qwen3

        wl_qk = use_weightless_qk_norm and not nope  # Llama 4 RoPE layers only
        self.q_wl_norm = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=False) if wl_qk else None
        self.k_wl_norm = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=False) if wl_qk else None

        # GPT-OSS: per-layer sliding window (even layers only) and attention sinks
        per_layer_sw = sliding_window if layer_idx % 2 == 0 else None

        if use_sinks:
            self.sinks = nn.Parameter(torch.zeros(self.num_heads))
            self.sinks.weight_loader = self._sinks_weight_loader
        else:
            self.sinks = None

        self.attn = Attention(
            self.num_heads, head_dim, head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
            sliding_window=per_layer_sw,
            sinks=self.sinks,
            attention_chunk_size=attention_chunk_size,
        )

    def _sinks_weight_loader(self, param, loaded_weight):
        """TP-shard attention sinks across heads."""
        rank = _tp_rank()
        heads_per_rank = param.data.size(0)
        start = rank * heads_per_rank
        param.data.copy_(loaded_weight.narrow(0, start, heads_per_rank))

    def _get_attn_scale(self, positions):  # Llama 4 NoPE only
        """Position-dependent attention temperature scaling."""
        floor = torch.floor((positions.float() + 1.0) / self.floor_scale)
        scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return scale.unsqueeze(-1)

    def forward(self, positions, hidden_states, rotary_emb=None):
        N = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Learnable QK norm (Qwen3: before RoPE)
        if self.q_norm is not None:
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(N, self.num_heads * self.head_dim)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(N, self.num_kv_heads * self.head_dim)

        rope = rotary_emb if rotary_emb is not None else self.rotary_emb
        if not self.nope and rope is not None:
            q, k = rope(positions, q, k)

        # Weight-less QK norm (Llama 4: after RoPE, only on RoPE layers)
        if self.q_wl_norm is not None:
            q = self.q_wl_norm(q.view(-1, self.head_dim)).view(N, -1)
            k = self.k_wl_norm(k.view(-1, self.head_dim)).view(N, -1)

        # Temperature tuning (Llama 4: only on NoPE layers)
        if self.attn_temperature_tuning:
            q = (q * self._get_attn_scale(positions)).to(q.dtype)

        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)


# Inlined from tasks/reference/L1/silu_and_mul.py
class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    @staticmethod
    def forward_cuda(x: torch.Tensor) -> torch.Tensor:
        return SiluAndMul.forward_native(x)

    def forward(self, x):
        return self.forward_native(x)


# Inlined from tasks/reference/L2/llama_mlp.py
class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config: dict | None = None,
                 hidden_size: int | None = None,
                 intermediate_size: int | None = None,
                 reduce_results: bool = True):
        super().__init__()
        h = hidden_size if hidden_size is not None else config.hidden_size
        i = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            h, [i] * 2,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            i, h,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)


# Inlined from tasks/reference/L3/llama_decoder.py
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 bias: bool = False, qk_norm: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            bias=bias, qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


from dataclasses import dataclass


@dataclass
class LlamaConfig:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 8.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position_embeddings: int = 8192
    dtype: torch.dtype = torch.bfloat16
    qkv_bias: bool = False

    @classmethod
    def from_pretrained(cls, model_name: str) -> "LlamaConfig":
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(model_name)
        # transformers 5.x moved rope config into rope_parameters dict
        rope_params = getattr(hf, "rope_parameters", None) or {}
        rope = getattr(hf, "rope_scaling", None) or {}
        # Merge: rope_parameters takes priority (transformers 5.x)
        rope = {**rope, **rope_params}
        rope_theta = rope.get("rope_theta") or getattr(hf, "rope_theta", 500000.0)
        is_qwen2 = getattr(hf, "model_type", "") in ("qwen2", "qwen2_moe")
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=rope.get("factor", 1.0),
            rope_low_freq_factor=rope.get("low_freq_factor", 1.0),
            rope_high_freq_factor=rope.get("high_freq_factor", 1.0),
            rope_original_max_position_embeddings=rope.get(
                "original_max_position_embeddings", hf.max_position_embeddings,
            ),
            qkv_bias=is_qwen2,
        )


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_low_freq_factor=config.rope_low_freq_factor,
            rope_high_freq_factor=config.rope_high_freq_factor,
            rope_original_max_position_embeddings=config.rope_original_max_position_embeddings,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, rotary_emb=self.rotary_emb,
                              bias=config.qkv_bias)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.capture_aux_hidden_states: bool = False
        # Capture indices follow sglang semantics: hidden_states + residual is
        # collected BEFORE each layer in this list runs. For Llama-3.1-8B
        # (L=32) sglang defaults to [2, num_layers//2, num_layers-3].
        self.aux_layer_ids: list[int] = []

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None
        aux_hidden_states: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            if self.capture_aux_hidden_states and i in self.aux_layer_ids:
                aux_hidden_states.append(
                    hidden_states if residual is None else hidden_states + residual
                )
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        if self.capture_aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def forward_with_lm_proj(self, input_ids, positions):
        """Forward pass including LM head linear (no gather)."""
        out = self.model(input_ids, positions)
        if isinstance(out, tuple):
            hidden_states, _ = out
        else:
            hidden_states = out
        return self.lm_head.project(hidden_states)

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None):
        """Enable capture of intermediate aux hidden states for EAGLE-3.

        With ``layer_ids=None`` defaults to sglang's heuristic
        ``[2, num_layers//2, num_layers-3]``. For Llama-3.1-8B this is
        ``[2, 16, 29]``.
        """
        num_layers = self.config.num_hidden_layers
        if layer_ids is None:
            layer_ids = [2, num_layers // 2, num_layers - 3]
        self.model.capture_aux_hidden_states = True
        self.model.aux_layer_ids = list(layer_ids)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, partial_logits):
        """Gather and finalize logits from pre-computed local partition."""
        logits = self.lm_head.gather_logits(partial_logits)
        if logits is not None:
            logits = logits.float()
        return logits

    def greedy_sample_decode(self, partial_logits):
        """Fast greedy sampling path: local argmax + small allgather."""
        result = self.lm_head.gather_greedy(partial_logits.float())
        return result
