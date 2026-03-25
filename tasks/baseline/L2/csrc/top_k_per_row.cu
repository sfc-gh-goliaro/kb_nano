// Adapted from vLLM (https://github.com/vllm-project/vllm)
// SPDX-License-Identifier: Apache-2.0
//
// Histogram-based per-row top-k selection with CUB radix sort fallback.
// Supports both prefill (per-row causal bounds via rowStarts/rowEnds) and
// decode (sequence-length-aware causal bounds).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <cfloat>
#include <cstdint>
#include <algorithm>

#define WARP_SIZE 32

namespace topk {

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      return bits & 0x3ff;
    }
  }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads,
                                   const T* in, idxT len, Func f) {
  constexpr int kWarpSize = WARP_SIZE;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
               sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(kWarpSize >= items_per_scalar);
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems,
          bool multipleBlocksPerRow, bool mergeBlocks, typename SmemFinalType,
          typename SmemOutputType>
__device__ bool processHistogramStep(
    const int* indices, const float* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx,
    int* smemFinalDstIdx, int* smemFinalBinSize, int* smemFoundTopKValues,
    SmemFinalType& smemFinal, int stride1, int rowStart, int topK) {
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }

  __syncthreads();

  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                   << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                    << patternShift;
  }

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemFinal.histo.data[binIdx], 1);
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                       rowEnd - rowStart, distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      distributeToBins(logit, idx);
    }
  }
  __syncthreads();

  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemFinal.histo.data[idx];

    __syncthreads();

    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

    prefixSum += lastValue;
    totalSum += lastValue;
    smemFinal.histo.data[idx] = prefixSum;

    __syncthreads();

    bool foundThreshold = false;
    if (prefixSum < topK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1
                              ? totalSum
                              : smemFinal.histo.data[idx + 1];

      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        foundThreshold = true;
      }
    }

    if (__syncthreads_or(foundThreshold)) {
      break;
    }

    lastValue = totalSum;
  }

  __syncthreads();

  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

        if constexpr (mergeBlocks) {
          smemOutput[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput[dstIdx] = idx + rowStart;
          reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
        } else {
          smemOutput[dstIdx] = idx;
        }
      }
      if constexpr (step < 3) {
        if (binIdx == thresholdBinIdx &&
            smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            if constexpr (mergeBlocks) {
              smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
              smemOutput[dstIdx] = idx + rowStart;
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
              smemOutput[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                       rowEnd - rowStart, processBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      processBins(logit, idx);
    }
  }

  __syncthreads();

  return smemFinalBinSize[0] > kNumFinalItems;
}

template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __device__ void topKPerRowJob(const int* indices, const float* logits,
                                     int rowStart, int rowEnd, int* outIndices,
                                     float* outLogits, int stride1, int topK) {
  static constexpr int kNumFinalItems = 2048;
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage =
      std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  struct FinalItems {
    int indices[kNumFinalItems];
    float logits[kNumFinalItems];
  };

  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smemFinal;

  extern __shared__ int32_t smemOutput[];

  __shared__ int smemThresholdBinIdx[1];
  __shared__ int smemFinalDstIdx[1];
  __shared__ int smemFinalBinSize[1];
  __shared__ int smemFoundTopKValues[1];

  int rowLen = rowEnd - rowStart;

  if (rowLen <= topK) {
    for (int rowIt = threadIdx.x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      if constexpr (multipleBlocksPerRow) {
        outIndices[rowIt] = rowIt + rowStart;
        outLogits[rowIt] = logits[rowIt + rowStart];
      } else {
        outIndices[rowIt] = rowIt;
      }
    }
    for (int rowIt = rowLen + threadIdx.x; rowIt < topK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
      if constexpr (multipleBlocksPerRow) {
        outLogits[rowIt] = -FLT_MAX;
      }
    }

    return;
  }
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  bool continueToNextStep =
      processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                           multipleBlocksPerRow, mergeBlocks>(
          indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
          smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
          smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

  if (continueToNextStep) {
    continueToNextStep =
        processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    continueToNextStep =
        processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                         multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
        smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
        smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (!continueToNextStep) {
    if constexpr (useRadixSort) {
      float finalLogits[kNumFinalItemsPerThread];
      int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        finalLogits[ii] = -FLT_MAX;
      }

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        if (srcIdx < smemFinalDstIdx[0]) {
          finalLogits[ii] = smemFinal.items.logits[srcIdx];
          finalIndices[ii] = smemFinal.items.indices[srcIdx];
        }
      }
      __syncthreads();

      FinalSort(smemFinal.finalSort)
          .SortDescendingBlockedToStriped(finalLogits, finalIndices);

      int baseIdx = smemFoundTopKValues[0];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        int dstIdx = baseIdx + srcIdx;

        if (dstIdx < topK) {
          smemOutput[dstIdx] = finalIndices[ii];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] =
                finalLogits[ii];
          }
        }
      }
    } else {
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx.x; i < smemFinalDstIdx[0];
           i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];
        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          if (logit < otherLogit || (logit == otherLogit && i < j)) {
            outIndex++;
          }
        }
        if (outIndex + baseIdx < topK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] =
                smemFinal.items.logits[i];
          }
        }
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock) {
    if constexpr (multipleBlocksPerRow) {
      outIndices[i] = smemOutput[i];
      outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
    } else {
      if (stride1 == 1) {
        outIndices[i] = smemOutput[i];
      } else {
        outIndices[i] = smemOutput[i] - rowStart;
      }
    }
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(
    const float* logits, const int* rowStarts, const int* rowEnds,
    int* outIndices, int stride0, int stride1, const int topK,
    const int offsetIndex) {
  static constexpr int kNumBins = 2048;

  int rowIdx = blockIdx.x + offsetIndex;

  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];

  outIndices += static_cast<int64_t>(rowIdx) * topK;
  logits += static_cast<int64_t>(rowIdx) * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
}

template <int kNumThreadsPerBlock, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(
    const float* logits, const int* seqLens, int* outIndices, int stride0,
    int stride1, const int topK, int next_n, float* outLogits = nullptr,
    const int numBlocksToMerge = 0, const int* indices = nullptr) {
  static constexpr int kNumBins = 2048;

  int rowIdx = blockIdx.x;

  int rowStart = 0;
  int seq_len = seqLens[rowIdx / next_n];
  int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  } else if constexpr (multipleBlocksPerRow) {
    const auto blockSize = rowEnd / gridDim.y;
    rowStart = blockSize * blockIdx.y;
    rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
    outIndices +=
        static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
    outLogits +=
        static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
  } else if constexpr (mergeBlocks) {
    rowEnd = numBlocksToMerge * topK;
    indices += static_cast<int64_t>(rowIdx) * numBlocksToMerge * topK;
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  }
  logits += static_cast<int64_t>(rowIdx) * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort,
                multipleBlocksPerRow, mergeBlocks>(
      indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK);
}

}  // namespace topk

// ---------------------------------------------------------------------------
// Host launcher functions
// ---------------------------------------------------------------------------

void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
                          const torch::Tensor& seqLens, torch::Tensor& indices,
                          int64_t numRows, int64_t stride0, int64_t stride1,
                          int64_t topK) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kSplitWorkThreshold = 200 * 1000;
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto numColumns = logits.size(1);

  if (numColumns < kSortingAlgorithmThreshold) {
    topk::topKPerRowDecode<kNumThreadsPerBlock, false>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(),
            indices.data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n));
  } else if (numColumns < kSplitWorkThreshold) {
    topk::topKPerRowDecode<kNumThreadsPerBlock, true>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(),
            indices.data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n));
  } else {
    constexpr auto multipleBlocksPerRowConfig = 10;

    const auto outIndicesAux =
        torch::empty({numRows, multipleBlocksPerRowConfig, topK},
                     torch::dtype(torch::kInt32).device(logits.device()));
    const auto outLogitsAux =
        torch::empty({numRows, multipleBlocksPerRowConfig, topK},
                     torch::dtype(torch::kFloat).device(logits.device()));

    topk::topKPerRowDecode<kNumThreadsPerBlock, true, true>
        <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerBlock,
           2 * topK * sizeof(int32_t), stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(),
            outIndicesAux.data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n), outLogitsAux.data_ptr<float>());

    constexpr int kNumThreadsPerBlockMerge = 1024;
    topk::topKPerRowDecode<kNumThreadsPerBlockMerge, true, false, true>
        <<<numRows, kNumThreadsPerBlockMerge, topK * sizeof(int32_t), stream>>>(
            outLogitsAux.data_ptr<float>(), seqLens.data_ptr<int>(),
            indices.data_ptr<int>(), multipleBlocksPerRowConfig * topK, 1,
            static_cast<int>(topK), static_cast<int>(next_n), nullptr,
            multipleBlocksPerRowConfig, outIndicesAux.data_ptr<int>());
  }
}

void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& rowStarts,
                           const torch::Tensor& rowEnds, torch::Tensor& indices,
                           int64_t numRows, int64_t stride0, int64_t stride1,
                           int64_t topK) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int numInsertionBlocks =
      std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
  topk::topKPerRowPrefill<kNumThreadsPerBlock, false>
      <<<numInsertionBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
         stream>>>(logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
                   rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
                   static_cast<int>(stride0), static_cast<int>(stride1),
                   static_cast<int>(topK), 0);

  if (numRows > kSortingAlgorithmThreshold) {
    int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
    topk::topKPerRowPrefill<kNumThreadsPerBlock, true>
        <<<numRadixBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
           stream>>>(logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
                     rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
                     static_cast<int>(stride0), static_cast<int>(stride1),
                     static_cast<int>(topK), kSortingAlgorithmThreshold);
  }
}

// ---------------------------------------------------------------------------
// PyBind11 module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("top_k_per_row_prefill", &top_k_per_row_prefill,
        "Per-row top-k with causal bounds (prefill)");
  m.def("top_k_per_row_decode", &top_k_per_row_decode,
        "Per-row top-k with sequence-length bounds (decode)");
}
