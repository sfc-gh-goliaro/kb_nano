// Standalone RMSNorm and fused-add-RMSNorm CUDA kernels for kb_nano.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "utils.h"

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
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float v = static_cast<float>(x[i]);
    sum_sq += v * v;
  }

  // warp reduce
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);

  // cross-warp reduce via shared memory
  __shared__ float shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  if (lane == 0) shared[wid] = sum_sq;
  __syncthreads();

  sum_sq = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
  if (wid == 0) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);
  }

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float v = static_cast<float>(x[i]);
    o[i] = static_cast<scalar_t>(v * s_rms_inv * static_cast<float>(weight[i]));
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
  dim3 block(std::min(hidden_size, 1024));
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

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);

  __shared__ float shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  if (lane == 0) shared[wid] = sum_sq;
  __syncthreads();

  sum_sq = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
  if (wid == 0) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);
  }

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  // Step 2: input = rmsnorm(residual) * weight
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float ri = static_cast<float>(r[i]);
    x[i] = static_cast<scalar_t>(ri * s_rms_inv * static_cast<float>(weight[i]));
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
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(input.scalar_type(), "fused_add_rmsnorm_kernel", [&] {
    fused_add_rmsnorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        hidden_size);
  });
}
