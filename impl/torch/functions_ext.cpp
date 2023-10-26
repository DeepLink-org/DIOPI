/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Generator.h>
#include <c10/util/Optional.h>
#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include "context.h"
#include "ext_kernel.h"
#include "functions_ext/flash-attention/include/flash_attn/flash_api.h"
#include "helper.hpp"

namespace {

c10::optional<at::Generator> buildGeneratorForMha(diopiContextHandle_t ctx, diopiGeneratorHandle_t gen, double dropoutP) {
    if (gen == nullptr) {
        if (dropoutP != 0) {
            throw std::runtime_error("dropout option requires a generator to be set");
        }
        return c10::nullopt;
    }
    return impl::aten::buildGenerator(ctx, gen);
}

}  // namespace

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
                                  diopiConstTensorHandle_t sin, const bool conj, const bool interleaved) {
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

diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                  diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRMS, diopiSize_t normalized_shape, double eps) {
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

diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                     double dropout_p, bool is_causal, bool return_debug_mask, double scale, diopiTensorHandle_t out,
                                     diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask) {
    impl::aten::setCurCtx(ctx);
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);
    at::Tensor atOutput, atLog_sumexp, atDebug_attn_mask, atQpaded, atKpaded, atVpaded, atOutpaded;
    auto atRng_state = at::empty({2}, at::kLong);
    // TORCH_CHECK(false, "There are currently cuda memory errors being returned from this path.")
    // Query -> Query (Batch x {Q_seq_len}  x Num_heads x Dim_per_head)
    // Key   -> Key   (Batch x {KV_seq_len} x Num_heads x Dim_per_head)
    // Value -> Value (Batch x {KV_seq_len} x Num_heads x Dim_per_head)
    // const int64_t batch_size = atQ.size(0);
    // const int64_t q_seq_len = atQ.size(1);
    // const int64_t num_heads = atQ.size(2);
    // const int64_t head_dim = atQ.size(3);
    atQ = atQ.contiguous();
    atK = atK.contiguous();
    atV = atV.contiguous();
    // std::vector<at::Tensor> mha_fwd(at::Tensor &q,                    // batch_size x seqlen_q x num_heads x head_size
    //                                 const at::Tensor &k,              // batch_size x seqlen_k x num_heads_k x head_size
    //                                 const at::Tensor &v,              // batch_size x seqlen_k x num_heads_k x head_size
    //                                 c10::optional<at::Tensor> &out_,  // batch_size x seqlen_q x num_heads x head_size
    //                                 const float p_dropout, const float softmax_scale, bool is_causal, const int window_size_left, int window_size_right,
    //                                 const bool return_softmax, c10::optional<at::Generator> gen_);
    // {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
    auto atOutputOpt = c10::optional<at::Tensor>(atOutput);
    std::vector<at::Tensor> result = mha_fwd(atQ, atK, atV, atOutputOpt, dropout_p, scale, is_causal, -1, -1, return_debug_mask, atGen);
    //(atOutput, atQpaded, atKpaded, atVpaded, atOutpaded, atLog_sumexp, atDebug_attn_mask, atRng_state)
    atOutput = result[0];
    atQpaded = result[1];
    atKpaded = result[2];
    atVpaded = result[3];
    atOutpaded = result[4];
    atLog_sumexp = result[5];
    atDebug_attn_mask = result[6];
    atRng_state = result[7];
    impl::aten::updateATen2Tensor(ctx, atOutput, out);
    // impl::aten::updateATen2Tensor(ctx, atLog_sumexp, softmax_lse);
    // impl::aten::updateATen2Tensor(ctx, atDebug_attn_mask, debug_attn_mask);

    // diopiTensorHandle_t new_state_handle = nullptr;
    // impl::aten::buildDiopiTensor(ctx, atRng_state, &new_state_handle);
    // diopiGeneratorSetState(gen, new_state_handle);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                             diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                             diopiConstTensorHandle_t softmax_lse, double dropout_p, bool is_causal, diopiGeneratorHandle_t gen, double scale,
                                             diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v) {
    impl::aten::setCurCtx(ctx);
    // at::Tensor atGrad_q, atGrad_k, atGrad_v;
    auto atGrad_q = impl::aten::buildATen(grad_q);
    auto atGrad_k = impl::aten::buildATen(grad_k);
    auto atGrad_v = impl::aten::buildATen(grad_v);
    // at::Tensor atGrad_softmax;
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);
    auto atGrad_out = impl::aten::buildATen(grad_out);
    auto atOut = impl::aten::buildATen(out);
    auto atLogsumexp = impl::aten::buildATen(softmax_lse);

    diopiTensorHandle_t* state_ptr = nullptr;
    diopiGeneratorGetState(ctx, gen, state_ptr);
    auto atState = impl::aten::buildATen(*state_ptr);
    int64_t atPhilox_seed = atState[0].item<int64_t>();
    int64_t atPhilox_offset = atState[1].item<int64_t>();
    // c10::optional<at::tensor>
    // c10::optional<at::Tensor> atState =
    // TORCH_CHECK(false, "There are currently cuda memory errors being returned from this path.")
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

    //(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor softmax_lse, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k,
    // float dropout_p, bool is_causal, int philox_seed, int philox_offset) -> (Tensor, Tensor, Tensor)

    // std::vector<at::Tensor>
    // mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
    //         const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
    //         const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
    //         const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
    //         const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
    //         const at::Tensor &softmax_lse,     // b x h x seqlen_q
    //         c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
    //         c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
    //         c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
    //         const float p_dropout,         // probability to drop
    //         const float softmax_scale,
    //         const bool is_causal,
    //         const int window_size_left,
    //         int window_size_right,
    //         c10::optional<at::Generator> gen_,
    //         c10::optional<at::Tensor> &rng_state)

    //     return { dq, dk, dv, softmax_d };
    auto atGrad_qOpt = c10::optional<at::Tensor>(atGrad_q);
    auto atGrad_kOpt = c10::optional<at::Tensor>(atGrad_k);
    auto atGrad_vOpt = c10::optional<at::Tensor>(atGrad_v);
    auto atStateOpt = c10::optional<at::Tensor>(atState);
    std::vector<at::Tensor> result =
        mha_bwd(atGrad_out, atQ, atK, atV, atOut, atLogsumexp, atGrad_qOpt, atGrad_kOpt, atGrad_vOpt, dropout_p, scale, is_causal, -1, -1, atGen, atStateOpt);
    //(atGrad_q, atGrad_k, atGrad_v, atGrad_softmax)
    atGrad_q = result[0];
    atGrad_k = result[1];
    atGrad_v = result[2];
    // at::Tensor atGrad_softmax = result[3];

    // atGrad_q = atGrad_q.view({batch_size, q_seq_len, num_heads, head_dim});
    // atGrad_k = atGrad_k.view({batch_size, q_seq_len, num_heads, head_dim});
    // atGrad_v = atGrad_v.view({batch_size, q_seq_len, num_heads, head_dim});

    impl::aten::updateATen2Tensor(ctx, atGrad_q, grad_q);
    impl::aten::updateATen2Tensor(ctx, atGrad_k, grad_k);
    impl::aten::updateATen2Tensor(ctx, atGrad_v, grad_v);
    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttentionVarLen(diopiContextHandle_t ctx, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                           diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k, int64_t max_q, int64_t max_k,
                                           double dropout_p, bool is_causal, bool return_debug_mask, double scale, diopiTensorHandle_t out,
                                           diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask) {
    impl::aten::setCurCtx(ctx);
    auto atQ = impl::aten::buildATen(q).clone();
    auto atK = impl::aten::buildATen(k).clone();
    auto atV = impl::aten::buildATen(v).clone();

    at::Tensor atOutput, atLog_sumexp, atDebug_attn_mask;
    int64_t atPhilox_seed{0}, atPhilox_offset{0};

    auto atCum_seq_q = impl::aten::buildATen(cum_seq_q);
    auto atCum_seq_k = impl::aten::buildATen(cum_seq_k);
    std::tie(atOutput, atLog_sumexp, atPhilox_seed, atPhilox_offset, atDebug_attn_mask) =
        at::_flash_attention_forward(atQ, atK, atV, atCum_seq_q, atCum_seq_k, max_q, max_k, dropout_p, is_causal, return_debug_mask);

    // 目前pytorch2.0版本返回的是(Tensor out, Tensor softmax_lse, int philox_seed, int philox_offset, Tensor debug_attn_mask)
    // 但是main分支的是返回的是五个tensor，因此存在一个转换的问题
    impl::aten::updateATen2Tensor(ctx, atOutput, out);
    impl::aten::updateATen2Tensor(ctx, atLog_sumexp, softmax_lse);
    impl::aten::updateATen2Tensor(ctx, atDebug_attn_mask, debug_attn_mask);

    at::Tensor new_state = at::empty({2}, at::kLong);
    // 可能存在bug，先测试
    new_state[0].fill_(atPhilox_seed);
    new_state[1].fill_(atPhilox_offset);

    diopiTensorHandle_t new_state_handle = nullptr;
    impl::aten::buildDiopiTensor(ctx, new_state, &new_state_handle);
    diopiGeneratorSetState(gen, new_state_handle);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttentionVarLenBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                                   diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                   diopiConstTensorHandle_t softmax_lse, diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k,
                                                   int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, diopiGeneratorHandle_t gen, double scale,
                                                   diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v) {
    impl::aten::setCurCtx(ctx);
    auto atGrad_q = impl::aten::buildATen(grad_q);
    auto atGrad_k = impl::aten::buildATen(grad_k);
    auto atGrad_v = impl::aten::buildATen(grad_v);

    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atGrad_out = impl::aten::buildATen(grad_out);
    auto atOut = impl::aten::buildATen(out);
    auto atLogsumexp = impl::aten::buildATen(softmax_lse);

    diopiTensorHandle_t* state_ptr = nullptr;
    diopiGeneratorGetState(ctx, gen, state_ptr);
    auto atState = impl::aten::buildATen(*state_ptr);
    int64_t atPhilox_seed = atState[0].item<int64_t>();
    int64_t atPhilox_offset = atState[1].item<int64_t>();

    auto atCum_seq_q = impl::aten::buildATen(cum_seq_q);
    auto atCum_seq_k = impl::aten::buildATen(cum_seq_k);
    std::tie(atGrad_q, atGrad_k, atGrad_v) = at::_flash_attention_backward(
        atGrad_out, atQ, atK, atV, atOut, atLogsumexp, atCum_seq_q, atCum_seq_k, max_q, max_k, dropout_p, is_causal, atPhilox_seed, atPhilox_offset);

    impl::aten::updateATen2Tensor(ctx, atGrad_q, grad_q);
    impl::aten::updateATen2Tensor(ctx, atGrad_k, grad_k);
    impl::aten::updateATen2Tensor(ctx, atGrad_v, grad_v);
    return diopiSuccess;
}

}  // extern "C"
