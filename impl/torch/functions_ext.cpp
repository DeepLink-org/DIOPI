/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/util/Optional.h>
#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include <cstdint>

// TODO(lljbash): the dependency on context.h makes no sense, check and refactor
#include "context.h"  // IWYU pragma: keep
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

void updateGeneratorStateForMha(diopiContextHandle_t ctx, diopiGeneratorHandle_t gen, const at::Tensor& atRngState) {
    if (gen == nullptr) {
        return;
    }
    auto cudaGen = at::cuda::detail::createCUDAGenerator();
    auto cudaGenImpl = at::check_generator<at::CUDAGeneratorImpl>(cudaGen);
    cudaGenImpl->set_current_seed(atRngState[0].item<std::int64_t>());
    cudaGenImpl->set_philox_offset_per_thread(atRngState[1].item<std::int64_t>());
    impl::aten::updateGeneratorHandleState(ctx, cudaGen, gen);
}

}  // namespace

extern "C" {

diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                  diopiConstTensorHandle_t sin, const bool conj, const bool interleaved) {
    impl::aten::setCurCtx(ctx);
    auto atX = impl::aten::buildATen(x);
    auto atCos = impl::aten::buildATen(cos);
    auto atSin = impl::aten::buildATen(sin);
    auto atOut = impl::aten::buildATen(out);
    int lastDim = atX.dim() - 1;          // 确定最后一个维度的索引
    auto chunks = atX.chunk(2, lastDim);  // 将 atX 切分为两个部分
    auto x1 = chunks[0];
    auto x2 = chunks[1];
    auto chunksOut = atOut.chunk(2, lastDim);
    auto out1 = chunksOut[0];
    auto out2 = chunksOut[1];
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
    auto atNormalizedShape = impl::aten::buildAtIntArray(normalized_shape);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);  // bias在这里实际上没有使用
    ext::ops::rms_norm_forward(atInput, atNormalizedShape, atWeight, eps, atOut, atInvRMS);
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
    auto atNormalizedShape = impl::aten::buildAtIntArray(normalized_shape);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);  // bias在这里实际上没有使用
    ext::ops::rms_norm_backward(atGradOutput, atInvRMS, atInput, atNormalizedShape, atWeight, eps, atGradInput, atGradWeight);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiTensorHandle_t q, diopiTensorHandle_t k, diopiTensorHandle_t v,
                                     double dropout_p, bool is_causal, bool return_debug_mask, double scale, diopiTensorHandle_t out,
                                     diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask) {
    impl::aten::setCurCtx(ctx);

    auto atQ = impl::aten::buildATen(q).contiguous();
    auto atK = impl::aten::buildATen(k).contiguous();
    auto atV = impl::aten::buildATen(v).contiguous();
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);

    c10::optional<at::Tensor> nullOpt;  // Workaround: flash_attn uses non-const optional& as args (which is a really bad idea)
    std::vector<at::Tensor> result = mha_fwd(atQ, atK, atV, nullOpt, dropout_p, scale, is_causal, -1, -1, return_debug_mask, atGen);
    const auto& atOutput = result[0];
    const auto& atQpaded = result[1];
    const auto& atKpaded = result[2];
    const auto& atVpaded = result[3];
    const auto& atOutpaded = result[4];
    const auto& atLogSumexp = result[5];
    const auto& atDebugAttnMask = result[6];
    const auto& atRngState = result[7];
    
    // const auto& atOutput = result[0];
    // const auto& atLogSumexp = result[5];
    // const auto& atDebugAttnMask = result[6];

    impl::aten::updateATen2Tensor(ctx, atQpaded, q);
    impl::aten::updateATen2Tensor(ctx, atKpaded, k);
    impl::aten::updateATen2Tensor(ctx, atVpaded, v);
    impl::aten::updateATen2Tensor(ctx, atOutpaded, out);
    // impl::aten::updateATen2Tensor(ctx, atOutput, out);
    impl::aten::updateATen2Tensor(ctx, atLogSumexp, softmax_lse);
    if (return_debug_mask) {
        impl::aten::updateATen2Tensor(ctx, atDebugAttnMask, debug_attn_mask);
    }

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                             diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                             diopiConstTensorHandle_t softmax_lse, double dropout_p, bool is_causal, diopiGeneratorHandle_t gen, double scale,
                                             diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v) {
    impl::aten::setCurCtx(ctx);

    auto atQ = impl::aten::buildATen(q).contiguous();
    auto atK = impl::aten::buildATen(k).contiguous();
    auto atV = impl::aten::buildATen(v).contiguous();
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);
    auto atGradOut = impl::aten::buildATen(grad_out).contiguous();
    auto atOut = impl::aten::buildATen(out).contiguous();
    auto atLogsumexp = impl::aten::buildATen(softmax_lse);

    c10::optional<at::Tensor> nullOpt;  // Workaround: flash_attn uses non-const optional& as args (which is a really bad idea)
    std::vector<at::Tensor> result =
        mha_bwd(atGradOut, atQ, atK, atV, atOut, atLogsumexp, nullOpt, nullOpt, nullOpt, dropout_p, scale, is_causal, -1, -1, atGen, nullOpt);
    const auto& atGradQ = result[0];
    const auto& atGradK = result[1];
    const auto& atGradV = result[2];

    impl::aten::updateATen2Tensor(ctx, atGradQ, grad_q);
    impl::aten::updateATen2Tensor(ctx, atGradK, grad_k);
    impl::aten::updateATen2Tensor(ctx, atGradV, grad_v);

    impl::aten::unsetCurCtx();
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
    auto atCum_seq_q = impl::aten::buildATen(cum_seq_q);
    auto atCum_seq_k = impl::aten::buildATen(cum_seq_k);
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);

    c10::optional<at::Tensor> outputNull;
    std::vector<at::Tensor> result =
        mha_varlen_fwd(atQ, atK, atV, outputNull, atCum_seq_q, atCum_seq_k, max_q, max_k, dropout_p, scale, false, is_causal, -1, -1, return_debug_mask, atGen);
    auto atOutput = result[0];
    auto atQ_padded = result[1];
    auto atKpaded = result[2];
    auto atVpaded = result[3];
    auto atOutpaded = result[4];
    auto atLogSumexp = result[5];
    auto atDebugAttnMask = result[6];
    auto atRngState = result[7];

    impl::aten::updateATen2Tensor(ctx, atOutput, out);
    impl::aten::updateATen2Tensor(ctx, atLogSumexp, softmax_lse);
    if (return_debug_mask) {
        impl::aten::updateATen2Tensor(ctx, atDebugAttnMask, debug_attn_mask);
    }
    updateGeneratorStateForMha(ctx, gen, atRngState);

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
    auto atQ = impl::aten::buildATen(q).contiguous();
    auto atK = impl::aten::buildATen(k).contiguous();
    auto atV = impl::aten::buildATen(v).contiguous();
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);
    auto atGrad_out = impl::aten::buildATen(grad_out).contiguous();
    auto atOut = impl::aten::buildATen(out).contiguous();
    auto atLogsumexp = impl::aten::buildATen(softmax_lse);
    auto atCum_seq_q = impl::aten::buildATen(cum_seq_q);
    auto atCum_seq_k = impl::aten::buildATen(cum_seq_k);
    diopiTensorHandle_t state_ptr = nullptr;
    diopiGeneratorGetState(ctx, gen, &state_ptr);
    auto atState = impl::aten::buildATen(state_ptr);

    auto atGrad_qOpt = c10::optional<at::Tensor>(atGrad_q);
    auto atGrad_kOpt = c10::optional<at::Tensor>(atGrad_k);
    auto atGrad_vOpt = c10::optional<at::Tensor>(atGrad_v);
    auto atStateOpt = c10::optional<at::Tensor>(atState);
    std::vector<at::Tensor> result = mha_varlen_bwd(atGrad_out,
                                                    atQ,
                                                    atK,
                                                    atV,
                                                    atOut,
                                                    atLogsumexp,
                                                    atGrad_qOpt,
                                                    atGrad_kOpt,
                                                    atGrad_vOpt,
                                                    atCum_seq_q,
                                                    atCum_seq_k,
                                                    max_q,
                                                    max_k,
                                                    dropout_p,
                                                    scale,
                                                    false,
                                                    is_causal,
                                                    -1,
                                                    -1,
                                                    atGen,
                                                    atStateOpt);
    atGrad_q = result[0];
    atGrad_k = result[1];
    atGrad_v = result[2];

    impl::aten::updateATen2Tensor(ctx, atGrad_q, grad_q);
    impl::aten::updateATen2Tensor(ctx, atGrad_k, grad_k);
    impl::aten::updateATen2Tensor(ctx, atGrad_v, grad_v);
    return diopiSuccess;
}

}  // extern "C"
