// kb_nano custom CUDA ops – PyTorch extension binding.
#include <torch/extension.h>

// Forward declarations
void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps);
void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps);
void silu_and_mul(at::Tensor& out, at::Tensor& input);
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);
void moe_sum(torch::Tensor& input, torch::Tensor& output);
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad, torch::Tensor cumsum_buffer,
                          bool pad_sorted_token_ids);
void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& gating_output, bool renormalize, double moe_softcapping,
                  const c10::optional<torch::Tensor>& correction_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA)");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add + RMSNorm (CUDA)");
  m.def("silu_and_mul", &silu_and_mul, "SiLU and Mul activation (CUDA)");
  m.def("rotary_embedding", &rotary_embedding, "Rotary position embedding (CUDA)");
  m.def("moe_sum", &moe_sum, "MoE sum reduction (CUDA)");
  m.def("moe_align_block_size", &moe_align_block_size, "MoE align block size (CUDA)");
  m.def("topk_softmax", &topk_softmax, "Top-K softmax for MoE (CUDA)");
}
