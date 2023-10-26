#pragma once
#ifndef FLASH_API_H_HZ4NF8VP
#define FLASH_API_H_HZ4NF8VP

#include <torch/extension.h>

std::vector<at::Tensor> mha_fwd(at::Tensor &q,                    // batch_size x seqlen_q x num_heads x head_size
                                const at::Tensor &k,              // batch_size x seqlen_k x num_heads_k x head_size
                                const at::Tensor &v,              // batch_size x seqlen_k x num_heads_k x head_size
                                c10::optional<at::Tensor> &out_,  // batch_size x seqlen_q x num_heads x head_size
                                const float p_dropout, const float softmax_scale, bool is_causal, const int window_size_left, int window_size_right,
                                const bool return_softmax, c10::optional<at::Generator> gen_);

std::vector<at::Tensor> mha_varlen_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &cu_seqlens_q,
                                       const at::Tensor &cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale,
                                       bool zero_tensors, bool is_causal, int window_size_left, int window_size_right, bool return_softmax, at::Generator gen);

std::vector<at::Tensor> mha_bwd(const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &out,
                                const at::Tensor &softmax_lse, c10::optional<at::Tensor> &dq_, c10::optional<at::Tensor> &dk_, c10::optional<at::Tensor> &dv_,
                                const float p_dropout, const float softmax_scale, const bool is_causal, const int window_size_left, int window_size_right,
                                c10::optional<at::Generator> gen_, c10::optional<at::Tensor> &rng_state);

std::vector<at::Tensor> mha_varlen_bwd(const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &out,
                                       const at::Tensor &softmax_lse, const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k, int max_seqlen_q,
                                       int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, bool is_causal, int window_size_left,
                                       int window_size_right, at::Generator gen);

std::vector<at::Tensor> mha_fwd_kvcache(at::Tensor &q, const at::Tensor &kcache, const at::Tensor &vcache, const at::Tensor &k, const at::Tensor &v,
                                        const at::Tensor &seqlens_k, const at::Tensor &rotary_cos, const at::Tensor &rotary_sin,
                                        const at::Tensor &cache_batch_idx, float softmax_scale, bool is_causal, int window_size_left, int window_size_right,
                                        bool is_rotary_interleaved, int num_splits);

#endif /* end of include guard: FLASH_API_H_HZ4NF8VP */
