/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include <iostream>

#include "context.h"
#include "ext_kernel.h"
#include "helper.hpp"

#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Generator.h>

/**
 * copyed from `aten/src/ATen/native/nested/cuda/NestedTensorTransformerFunctions.cpp`
 * This function is used to calculate two pieces of metadata that are needed
 * for use with flash-attention and efficient_attention kernels. They are the
 * cumulative sequence_length over a batch of sequences and the maximum sequence
 * length.
 *
 * @return A tuple of cumulative sequence lengths and the maximum sequence length,
 * and the last element in the cumulative_sequence_lengths
 */
std::tuple<at::Tensor, int64_t, int64_t> cumulative_and_max_seq_len(at::Tensor qkv) {
    TORCH_CHECK(qkv.is_nested(), "QKV must be nested for flash cumulative_seq_len calculation.")
    auto* nt_impl = at::native::get_nested_tensor_impl(qkv);
    const auto& sizes = nt_impl->get_nested_size_tensor();
    auto size_tensor_stride = sizes.stride(0);

    const int64_t batch_size = qkv.size(0);
    auto cumulative_seqlen = at::zeros({batch_size + 1}, c10::TensorOptions().device(at::kCPU).dtype(at::kInt));

    auto* sizes_ptr = sizes.data_ptr<int64_t>();
    auto* cumulative_seqlen_ptr = cumulative_seqlen.data_ptr<int32_t>();

    int32_t sum = 0;
    int64_t max_seqlen = -1;
    cumulative_seqlen_ptr[0] = sum;
    for (const auto i : c10::irange(batch_size)) {
        // Calculate the cumulative sum of the sequence lengths
        auto current_seq_len = sizes_ptr[(i * size_tensor_stride)];
        sum += current_seq_len;
        cumulative_seqlen_ptr[i + 1] = sum;

        // Find the max element while we traverse
        max_seqlen = std::max(max_seqlen, current_seq_len);
    }
    // Send to GPU, this is pretty light weight calc for normal batch size
    // but maybe this needs to be on gpu
    cumulative_seqlen = cumulative_seqlen.to(c10::TensorOptions().device(at::kCUDA));
    return std::tuple<at::Tensor, int64_t, int64_t>{cumulative_seqlen, max_seqlen, sum};
}

extern "C" {

diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                  diopiConstTensorHandle_t sin, const bool conj) {
    impl::aten::setCurCtx(ctx);
    auto atX = impl::aten::buildATen(x);
    auto atCos = impl::aten::buildATen(cos);
    auto atSin = impl::aten::buildATen(sin);
    auto atOut = impl::aten::buildATen(out);
    int last_dim = atX.dim() - 1;          // 确定最后一个维度的索引
    auto chunks = atX.chunk(2, last_dim);  // 将 atX 切分为两个部分
    auto x1 = chunks[0];
    auto x2 = chunks[1];
    auto chunks_out = atOut.chunk(2, last_dim);
    auto out1 = chunks_out[0];
    auto out2 = chunks_out[1];
    ext::ops::apply_rotary_cuda(x1, x2, atCos, atSin, out1, out2, conj);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRMS, diopiConstTensorHandle_t input,
                          diopiSize_t normalized_shape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInvRMS = impl::aten::buildATen(invRMS);
    auto atInput = impl::aten::buildATen(input);
    auto atNormalized_shape = impl::aten::buildAtIntArray(normalized_shape);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);  // bias在这里实际上没有使用
    ext::ops::rms_norm_forward(atInput, atNormalized_shape, atWeight, eps, atOut, atInvRMS);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                            diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRMS,
                                            diopiSize_t normalized_shape, double eps) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(gradInput);
    auto atGradWeight = impl::aten::buildATen(gradWeight);
    auto atGradBias = impl::aten::buildATen(gradBias);
    auto atGradOutput = impl::aten::buildATen(gradOutput);
    auto atInvRMS = impl::aten::buildATen(invRMS);
    auto atInput = impl::aten::buildATen(input);
    auto atNormalized_shape = impl::aten::buildAtIntArray(normalized_shape);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);  // bias在这里实际上没有使用
    ext::ops::rms_norm_backward(atGradOutput, atInvRMS, atInput, atNormalized_shape, atWeight, eps, atGradInput, atGradWeight);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k, int64_t* max_q, int64_t* max_k,
                                               double dropout_p, bool is_causal, bool return_debug_mask, double* scale, diopiTensorHandle_t output,
                                               diopiTensorHandle_t softmax_logsumexp, diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask) {
    impl::aten::setCurCtx(ctx);
    auto atQ = impl::aten::buildATen(q).clone();
    auto atK = impl::aten::buildATen(k).clone();
    auto atV = impl::aten::buildATen(v).clone();

    at::Tensor atOutput, atLog_sumexp, atDebug_attn_mask;
    uint64_t atPhilox_seed{0}, atPhilox_offset{0};
    if (max_q == nullptr && max_k == nullptr && cum_seq_q == nullptr && cum_seq_q == nullptr) {
        TORCH_CHECK(false, "There are currently cuda memory errors being returned from this path.")
        // Query -> Query (Batch x {Q_seq_len}  x Num_heads x Dim_per_head)
        // Key   -> Key   (Batch x {KV_seq_len} x Num_heads x Dim_per_head)
        // Value -> Value (Batch x {KV_seq_len} x Num_heads x Dim_per_head)
        const int64_t batch_size = atQ.size(0);
        const int64_t q_seq_len = atQ.size(1);
        const int64_t num_heads = atQ.size(2);
        const int64_t head_dim = atQ.size(3);

        atQ = atQ.contiguous();
        atK = atK.contiguous();
        atV = atV.contiguous();

        // K and V have to have the same Nnz, should probably torch_check
        // assume in order to not iterate over v

        auto cumulative_and_max_q = cumulative_and_max_seq_len(atQ);
        auto cumulative_and_max_k = cumulative_and_max_seq_len(atK);

        at::Tensor cumulative_sequence_length_q = std::get<0>(cumulative_and_max_q);
        at::Tensor cumulative_sequence_length_k = std::get<0>(cumulative_and_max_k);

        const int64_t max_seqlen_batch_q = std::get<1>(cumulative_and_max_q);
        const int64_t max_seqlen_batch_k = std::get<1>(cumulative_and_max_k);

        const int64_t Nnz_q = cumulative_sequence_length_q[-1].item<int64_t>();
        const int64_t Nnz_kv = cumulative_sequence_length_k[-1].item<int64_t>();

        auto query_buffer_reshaped = atQ.view({Nnz_q, num_heads, head_dim});
        auto key_buffer_reshaped = atK.view({Nnz_kv, num_heads, head_dim});
        auto value_buffer_reshaped = atV.view({Nnz_kv, num_heads, head_dim});

        std::tie(atOutput, atLog_sumexp, atPhilox_seed, atPhilox_offset, atDebug_attn_mask) = at::_flash_attention_forward(query_buffer_reshaped,
                                                                                                                           key_buffer_reshaped,
                                                                                                                           value_buffer_reshaped,
                                                                                                                           cumulative_sequence_length_q,
                                                                                                                           cumulative_sequence_length_k,
                                                                                                                           max_seqlen_batch_q,
                                                                                                                           max_seqlen_batch_k,
                                                                                                                           dropout_p,
                                                                                                                           is_causal,
                                                                                                                           return_debug_mask);
        atOutput = atOutput.view({batch_size, q_seq_len, num_heads, head_dim});
    } else {
        auto atCum_seq_q = impl::aten::buildATen(cum_seq_q);
        auto atCum_seq_k = impl::aten::buildATen(cum_seq_k);
        std::tie(atOutput, atLog_sumexp, atPhilox_seed, atPhilox_offset, atDebug_attn_mask) =
            at::_flash_attention_forward(atQ, atK, atV, atCum_seq_q, atCum_seq_k, *max_q, *max_k, dropout_p, is_causal, return_debug_mask);
    }

    // 目前pytorch2.0版本返回的是(Tensor output, Tensor softmax_logsumexp, int philox_seed, int philox_offset, Tensor debug_attn_mask)
    // 但是main分支的是返回的是五个tensor，因此存在一个转换的问题
    impl::aten::updateATen2Tensor(ctx, atOutput, output);
    impl::aten::updateATen2Tensor(ctx, atLog_sumexp, softmax_logsumexp);
    impl::aten::updateATen2Tensor(ctx, atDebug_attn_mask, debug_attn_mask);

    at::Tensor new_state = at::empty({2});
    // 可能存在bug，先测试
    new_state[0].fill_(static_cast<double>(atPhilox_seed));;
    new_state[1].fill_(static_cast<double>(atPhilox_offset));

    diopiTensorHandle_t new_state_handle = nullptr;
    impl::aten::buildDiopiTensor(ctx, new_state, &new_state_handle);
    diopiGeneratorSetState(gen, new_state_handle);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                                       diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                       diopiConstTensorHandle_t logsumexp, diopiConstTensorHandle_t cum_seq_q,
                                                       diopiConstTensorHandle_t cum_seq_k, int64_t* max_q, int64_t* max_k, double dropout_p, bool is_causal,
                                                       diopiGeneratorHandle_t gen, double* scale, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k,
                                                       diopiTensorHandle_t grad_v) {
    impl::aten::setCurCtx(ctx);
    auto atGrad_q = impl::aten::buildATen(grad_q);
    auto atGrad_k = impl::aten::buildATen(grad_k);
    auto atGrad_v = impl::aten::buildATen(grad_v);

    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atGrad_out = impl::aten::buildATen(grad_out);
    auto atOut = impl::aten::buildATen(out);
    auto atLogsumexp = impl::aten::buildATen(logsumexp);

    diopiTensorHandle_t* state_ptr = nullptr;
    diopiGeneratorGetState(ctx, gen, state_ptr);
    auto atState = impl::aten::buildATen(*state_ptr);
    uint64_t atPhilox_seed = atState[0].item<uint64_t>();
    uint64_t atPhilox_offset = atState[1].item<uint64_t>();

    if (max_q == nullptr && max_k == nullptr && cum_seq_q == nullptr && cum_seq_q == nullptr) {
        TORCH_CHECK(false, "There are currently cuda memory errors being returned from this path.")
        // Query -> Query (Batch x {Q_seq_len}  x Num_heads x Dim_per_head)
        // Key   -> Key   (Batch x {KV_seq_len} x Num_heads x Dim_per_head)
        // Value -> Value (Batch x {KV_seq_len} x Num_heads x Dim_per_head)
        const int64_t batch_size = atQ.size(0);
        const int64_t q_seq_len = atQ.size(1);
        const int64_t num_heads = atQ.size(2);
        const int64_t head_dim = atQ.size(3);

        atQ = atQ.contiguous();
        atK = atK.contiguous();
        atV = atV.contiguous();
        atGrad_out = atGrad_out.contiguous();
        atOut = atOut.contiguous();

        // K and V have to have the same Nnz, should probably torch_check
        // assume in order to not iterate over v

        auto cumulative_and_max_q = cumulative_and_max_seq_len(atQ);
        auto cumulative_and_max_k = cumulative_and_max_seq_len(atK);

        at::Tensor cumulative_sequence_length_q = std::get<0>(cumulative_and_max_q);
        at::Tensor cumulative_sequence_length_k = std::get<0>(cumulative_and_max_k);

        const int64_t max_seqlen_batch_q = std::get<1>(cumulative_and_max_q);
        const int64_t max_seqlen_batch_k = std::get<1>(cumulative_and_max_k);

        const int64_t Nnz_q = cumulative_sequence_length_q[-1].item<int64_t>();
        const int64_t Nnz_kv = cumulative_sequence_length_k[-1].item<int64_t>();

        auto grad_query_buffer_reshaped = atGrad_out.view({Nnz_q, num_heads, head_dim});
        auto query_buffer_reshaped = atQ.view({Nnz_q, num_heads, head_dim});
        auto key_buffer_reshaped = atK.view({Nnz_kv, num_heads, head_dim});
        auto value_buffer_reshaped = atV.view({Nnz_kv, num_heads, head_dim});
        auto out_reshaped = atOut.view({Nnz_kv, num_heads, head_dim});
        auto lse_reshaped = atLogsumexp.view({Nnz_kv, num_heads, head_dim});

        //(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k,
        // float dropout_p, bool is_causal, int philox_seed, int philox_offset) -> (Tensor, Tensor, Tensor)

        std::tie(atGrad_q, atGrad_k, atGrad_v) = at::_flash_attention_backward(grad_query_buffer_reshaped,
                                                                               query_buffer_reshaped,
                                                                               key_buffer_reshaped,
                                                                               value_buffer_reshaped,
                                                                               out_reshaped,
                                                                               lse_reshaped,
                                                                               cumulative_sequence_length_q,
                                                                               cumulative_sequence_length_k,
                                                                               max_seqlen_batch_q,
                                                                               max_seqlen_batch_k,
                                                                               dropout_p,
                                                                               is_causal,
                                                                               atPhilox_seed,
                                                                               atPhilox_offset);
        atGrad_q = atGrad_q.view({batch_size, q_seq_len, num_heads, head_dim});
        atGrad_k = atGrad_k.view({batch_size, q_seq_len, num_heads, head_dim});
        atGrad_v = atGrad_v.view({batch_size, q_seq_len, num_heads, head_dim});
    } else {
        auto atCum_seq_q = impl::aten::buildATen(cum_seq_q);
        auto atCum_seq_k = impl::aten::buildATen(cum_seq_k);
        std::tie(atGrad_q, atGrad_k, atGrad_v) = at::_flash_attention_backward(
            atGrad_out, atQ, atK, atV, atOut, atLogsumexp, atCum_seq_q, atCum_seq_k, *max_q, *max_k, dropout_p,is_causal, atPhilox_seed, atPhilox_offset);
    }

    impl::aten::updateATen2Tensor(ctx, atGrad_q, grad_q);
    impl::aten::updateATen2Tensor(ctx, atGrad_k, grad_k);
    impl::aten::updateATen2Tensor(ctx, atGrad_v, grad_v);
    return diopiSuccess;
}

}  // extern "C"
