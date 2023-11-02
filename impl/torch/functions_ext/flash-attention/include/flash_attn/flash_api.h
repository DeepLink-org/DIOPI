#pragma once
#ifndef FLASH_API_H_HZ4NF8VP
#define FLASH_API_H_HZ4NF8VP

#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> mha_fwd(at::Tensor &q,                    // batch_size x seqlen_q x num_heads x head_size
                                const at::Tensor &k,              // batch_size x seqlen_k x num_heads_k x head_size
                                const at::Tensor &v,              // batch_size x seqlen_k x num_heads_k x head_size
                                c10::optional<at::Tensor> &out_,  // batch_size x seqlen_q x num_heads x head_size
                                const float p_dropout, const float softmax_scale, bool is_causal, const int window_size_left, int window_size_right,
                                const bool return_softmax, c10::optional<at::Generator> gen_);

std::vector<at::Tensor> mha_varlen_fwd(const at::Tensor &q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       const at::Tensor &k,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const at::Tensor &v,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                       c10::optional<at::Tensor> &out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const at::Tensor &cu_seqlens_q,   // b+1
                                       const at::Tensor &cu_seqlens_k,   // b+1
                                       const int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale,
                                       const bool zero_tensors, const bool is_causal, const int window_size_left, int window_size_right,
                                       const bool return_softmax, c10::optional<at::Generator> gen_);

std::vector<at::Tensor> mha_bwd(const at::Tensor &dout,          // batch_size x seqlen_q x num_heads, x head_size_og
                                const at::Tensor &q,             // batch_size x seqlen_q x num_heads x head_size
                                const at::Tensor &k,             // batch_size x seqlen_k x num_heads_k x head_size
                                const at::Tensor &v,             // batch_size x seqlen_k x num_heads_k x head_size
                                const at::Tensor &out,           // batch_size x seqlen_q x num_heads x head_size
                                const at::Tensor &softmax_lse,   // b x h x seqlen_q
                                c10::optional<at::Tensor> &dq_,  // batch_size x seqlen_q x num_heads x head_size
                                c10::optional<at::Tensor> &dk_,  // batch_size x seqlen_k x num_heads_k x head_size
                                c10::optional<at::Tensor> &dv_,  // batch_size x seqlen_k x num_heads_k x head_size
                                const float p_dropout,           // probability to drop
                                const float softmax_scale, const bool is_causal, const int window_size_left, int window_size_right,
                                c10::optional<at::Generator> gen_, c10::optional<at::Tensor> &rng_state);

std::vector<at::Tensor> mha_varlen_bwd(const at::Tensor &dout,          // total_q x num_heads, x head_size
                                       const at::Tensor &q,             // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       const at::Tensor &k,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const at::Tensor &v,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const at::Tensor &out,           // total_q x num_heads x head_size
                                       const at::Tensor &softmax_lse,   // b x h x s   softmax logsumexp
                                       c10::optional<at::Tensor> &dq_,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       c10::optional<at::Tensor> &dk_,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                       c10::optional<at::Tensor> &dv_,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const at::Tensor &cu_seqlens_q,  // b+1
                                       const at::Tensor &cu_seqlens_k,  // b+1
                                       const int max_seqlen_q,
                                       const int max_seqlen_k,  // max sequence length to choose the kernel
                                       const float p_dropout,   // probability to drop
                                       const float softmax_scale, const bool zero_tensors, const bool is_causal, const int window_size_left,
                                       int window_size_right, c10::optional<at::Generator> gen_, c10::optional<at::Tensor> &rng_state);

std::vector<at::Tensor> mha_fwd_kvcache(at::Tensor &q, const at::Tensor &kcache, const at::Tensor &vcache, const at::Tensor &k, const at::Tensor &v,
                                        const at::Tensor &seqlens_k, const at::Tensor &rotary_cos, const at::Tensor &rotary_sin,
                                        const at::Tensor &cache_batch_idx, float softmax_scale, bool is_causal, int window_size_left, int window_size_right,
                                        bool is_rotary_interleaved, int num_splits);

#endif /* end of include guard: FLASH_API_H_HZ4NF8VP */
