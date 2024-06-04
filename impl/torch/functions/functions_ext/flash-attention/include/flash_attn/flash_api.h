// Declare functions implemented in flash-attention library (v2.3.x).
//
// WARNING: The flash-attention library exports these functions in global namespace. Bad practice. Nothing we can do about it.

#ifndef IMPL_TORCH_FUNCTIONS_FUNCTIONS_EXT_FLASH_ATTENTION_INCLUDE_FLASH_ATTN_FLASH_API_H_
#define IMPL_TORCH_FUNCTIONS_FUNCTIONS_EXT_FLASH_ATTENTION_INCLUDE_FLASH_ATTN_FLASH_API_H_

#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/Optional.h>

#include <stdexcept>  // IWYU pragma: keep
#include <vector>

#define DIOPI_EXT_CALL_FLASH(func, ...)                                                                         \
    [&] {                                                                                                       \
        if (func == nullptr) {                                                                                  \
            throw std::runtime_error("unable to call flash " #func ": DIOPI is built without flash-attention"); \
        }                                                                                                       \
        return func(__VA_ARGS__);                                                                               \
    }()

// use these indices to access the return values of the functions
namespace impl::cuda::fa {
// mha_fwd and mha_varlen_fwd share the same return indices
namespace mha_fwd_ret_idx {
enum {
    OUT = 0,
    Q_PADDED,
    K_PADDED,
    V_PADDED,
    OUT_PADDED,
    SOFTMAX_LSE,
    DEBUG_ATTN_MASK,
    RNG_STATE,
};
}  // namespace mha_fwd_ret_idx
}  // namespace impl::cuda::fa

std::vector<at::Tensor> __attribute__((weak)) mha_fwd(at::Tensor& q,                    // batch_size x seqlen_q x num_heads x head_size
                                                      const at::Tensor& k,              // batch_size x seqlen_k x num_heads_k x head_size
                                                      const at::Tensor& v,              // batch_size x seqlen_k x num_heads_k x head_size
                                                      c10::optional<at::Tensor>& out_,  // batch_size x seqlen_q x num_heads x head_size
                                                      const float p_dropout, const float softmax_scale, bool is_causal, const int window_size_left,
                                                      int window_size_right, const bool return_softmax, c10::optional<at::Generator> gen_);

std::vector<at::Tensor> __attribute__((weak)) mha_varlen_fwd(const at::Tensor& q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                                             const at::Tensor& k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                                             const at::Tensor& v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                                             c10::optional<at::Tensor>& out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                                             const at::Tensor& cu_seqlens_q,   // b+1
                                                             const at::Tensor& cu_seqlens_k,   // b+1
                                                             const int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale,
                                                             const bool zero_tensors, const bool is_causal, const int window_size_left, int window_size_right,
                                                             const bool return_softmax, c10::optional<at::Generator> gen_);

std::vector<at::Tensor> __attribute__((weak)) mha_bwd(const at::Tensor& dout,          // batch_size x seqlen_q x num_heads, x head_size_og
                                                      const at::Tensor& q,             // batch_size x seqlen_q x num_heads x head_size
                                                      const at::Tensor& k,             // batch_size x seqlen_k x num_heads_k x head_size
                                                      const at::Tensor& v,             // batch_size x seqlen_k x num_heads_k x head_size
                                                      const at::Tensor& out,           // batch_size x seqlen_q x num_heads x head_size
                                                      const at::Tensor& softmax_lse,   // b x h x seqlen_q
                                                      c10::optional<at::Tensor>& dq_,  // batch_size x seqlen_q x num_heads x head_size
                                                      c10::optional<at::Tensor>& dk_,  // batch_size x seqlen_k x num_heads_k x head_size
                                                      c10::optional<at::Tensor>& dv_,  // batch_size x seqlen_k x num_heads_k x head_size
                                                      const float p_dropout,           // probability to drop
                                                      const float softmax_scale, const bool is_causal, const int window_size_left, int window_size_right,
                                                      c10::optional<at::Generator> gen_, c10::optional<at::Tensor>& rng_state);

std::vector<at::Tensor> __attribute__((weak)) mha_varlen_bwd(
    const at::Tensor& dout,          // total_q x num_heads, x head_size
    const at::Tensor& q,             // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& v,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& out,           // total_q x num_heads x head_size
    const at::Tensor& softmax_lse,   // b x h x s   softmax logsumexp
    c10::optional<at::Tensor>& dq_,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    c10::optional<at::Tensor>& dk_,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    c10::optional<at::Tensor>& dv_,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1
    const int max_seqlen_q,
    const int max_seqlen_k,  // max sequence length to choose the kernel
    const float p_dropout,   // probability to drop
    const float softmax_scale, const bool zero_tensors, const bool is_causal, const int window_size_left, int window_size_right,
    c10::optional<at::Generator> gen_, c10::optional<at::Tensor>& rng_state);

std::vector<at::Tensor> __attribute__((weak)) mha_fwd_kvcache(at::Tensor& q, const at::Tensor& kcache, const at::Tensor& vcache, const at::Tensor& k,
                                                              const at::Tensor& v, const at::Tensor& seqlens_k, const at::Tensor& rotary_cos,
                                                              const at::Tensor& rotary_sin, const at::Tensor& cache_batch_idx, float softmax_scale,
                                                              bool is_causal, int window_size_left, int window_size_right, bool is_rotary_interleaved,
                                                              int num_splits);

#endif  // IMPL_TORCH_FUNCTIONS_FUNCTIONS_EXT_FLASH_ATTENTION_INCLUDE_FLASH_ATTN_FLASH_API_H_
