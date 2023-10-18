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
                                               diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k, int max_q, int max_k, float dropout_p,
                                               bool is_causal, bool return_debug_mask, diopiTensorHandle_t output, diopiTensorHandle_t softmax_logsumexp,
                                               int& philox_seed, int& philox_offset, diopiTensorHandle_t debug_attn_mask) {
    impl::aten::setCurCtx(ctx);
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atCum_seq_q = impl::aten::buildATen(cum_seq_q);
    auto atCum_seq_k = impl::aten::buildATen(cum_seq_k);
    auto OutRes = at::_flash_attention_forward(atQ, atK, atV, atCum_seq_q, atCum_seq_k, max_q, max_k, dropout_p, is_causal, return_debug_mask);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(OutRes), output);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(OutRes), softmax_logsumexp);
    impl::aten::updateATen2Tensor(ctx, std::get<4>(OutRes), debug_attn_mask);
    philox_seed = std::get<2>(OutRes);
    philox_offset = std::get<3>(OutRes);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                                       diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                       diopiConstTensorHandle_t logsumexp, diopiConstTensorHandle_t cum_seq_q,
                                                       diopiConstTensorHandle_t cum_seq_k, int max_q, int max_k, float dropout_p, bool is_causal,
                                                       int philox_seed, int philox_offset, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k,
                                                       diopiTensorHandle_t grad_v) {
    impl::aten::setCurCtx(ctx);
    auto atGrad_out = impl::aten::buildATen(grad_out);
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atOut = impl::aten::buildATen(out);
    auto atLogsumexp = impl::aten::buildATen(logsumexp);
    auto atCum_seq_q = impl::aten::buildATen(cum_seq_q);
    auto atCum_seq_k = impl::aten::buildATen(cum_seq_k);
    auto OutRes = at::_flash_attention_backward(
        atGrad_out, atQ, atK, atV, atOut, atLogsumexp, atCum_seq_q, atCum_seq_k, max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(OutRes), grad_q);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(OutRes), grad_k);
    impl::aten::updateATen2Tensor(ctx, std::get<2>(OutRes), grad_v);
    return diopiSuccess;
}

}  // extern "C"
