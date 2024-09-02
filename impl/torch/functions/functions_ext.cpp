/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/sum.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_ext.h>
#include <flash_attn/flash_api.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../ext_kernel.h"
#include "../helper.hpp"

namespace impl::cuda {

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

diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                  diopiConstTensorHandle_t sin, const bool conj, const bool interleaved) {
    if (interleaved) {
        set_last_error_string("interleaved rotary embedding is not supported yet");
        return diopiNoImplement;
    }
    impl::aten::setCurStream(ctx);
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

    return diopiSuccess;
}

diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRMS, diopiConstTensorHandle_t input,
                          diopiSize_t normalized_shape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    impl::aten::setCurStream(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInvRMS = impl::aten::buildATen(invRMS);
    auto atInput = impl::aten::buildATen(input);
    auto atNormalizedShape = impl::aten::buildAtIntArray(normalized_shape);
    auto atWeight = impl::aten::buildATen(weight);
    ext::ops::rms_norm_forward(atInput, atNormalizedShape, atWeight, eps, atOut, atInvRMS);
    if (bias) {
        auto atBias = impl::aten::buildATen(bias);
        atOut.add_(atBias);
    }
    return diopiSuccess;
}

diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                  diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRMS, diopiSize_t normalized_shape, double eps) {
    impl::aten::setCurStream(ctx);
    auto atGradInput = impl::aten::buildATen(gradInput);
    auto atGradWeight = impl::aten::buildATen(gradWeight);
    auto atGradOutput = impl::aten::buildATen(gradOutput);
    auto atInvRMS = impl::aten::buildATen(invRMS);
    auto atInput = impl::aten::buildATen(input);
    auto atNormalizedShape = impl::aten::buildAtIntArray(normalized_shape);
    auto atWeight = impl::aten::buildATen(weight);
    ext::ops::rms_norm_backward(atGradOutput, atInvRMS, atInput, atNormalizedShape, atWeight, eps, atGradInput, atGradWeight);
    if (gradBias) {
        auto atGradBias = impl::aten::buildATen(gradBias);
        auto outDim = atGradOutput.dim();
        auto biasDim = atGradBias.dim();
        if (outDim > biasDim) {
            std::vector<int64_t> sumDims(outDim - biasDim);
            std::iota(sumDims.begin(), sumDims.end(), 0);
            at::sum_out(atGradBias, atGradOutput, sumDims);
        } else {
            atGradBias.copy_(atGradOutput);
        }
    }
    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiTensorHandle_t q, diopiTensorHandle_t k, diopiTensorHandle_t v, double dropout_p,
                                     bool is_causal, bool return_debug_mask, double scale, diopiTensorHandle_t out, diopiTensorHandle_t softmax_lse,
                                     diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask) {
    impl::aten::setCurStream(ctx);

    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optOut, out);
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);

    auto headSize = atQ.sizes()[3];
    TORCH_CHECK(headSize % 8 == 0, "DIOPI now only support head sizes which are multiple of 8");

    c10::optional<at::Tensor> nullOpt;
    std::vector<at::Tensor> result =
        DIOPI_EXT_CALL_FLASH(mha_fwd, atQ, atK, atV, optOut, nullOpt, dropout_p, scale, is_causal, -1, -1, return_debug_mask, atGen);

    // PERF: these copy can be eliminated by modifying the flash_attn api
    impl::aten::updateATen2Tensor(ctx, result[fa::mha_fwd_ret_idx::SOFTMAX_LSE], softmax_lse);
    if (return_debug_mask) {
        impl::aten::updateATen2Tensor(ctx, result[fa::mha_fwd_ret_idx::DEBUG_ATTN_MASK], debug_attn_mask);
    }

    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                             diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                             diopiConstTensorHandle_t softmax_lse, double dropout_p, bool is_causal, diopiGeneratorHandle_t gen, double scale,
                                             diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v) {
    impl::aten::setCurStream(ctx);

    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);
    auto atGradOut = impl::aten::buildATen(grad_out);
    auto atOut = impl::aten::buildATen(out);
    auto atLogsumexp = impl::aten::buildATen(softmax_lse);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradQ, grad_q);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradK, grad_k);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradV, grad_v);
    c10::optional<at::Tensor> nullStateOpt;  // Workaround: flash_attn uses non-const optional& as args (which is a really bad idea)
    c10::optional<at::Tensor> nullSlopesOpt;

    std::vector<at::Tensor> result = DIOPI_EXT_CALL_FLASH(mha_bwd,
                                                          atGradOut,
                                                          atQ,
                                                          atK,
                                                          atV,
                                                          atOut,
                                                          atLogsumexp,
                                                          optGradQ,
                                                          optGradK,
                                                          optGradV,
                                                          nullSlopesOpt,
                                                          dropout_p,
                                                          scale,
                                                          is_causal,
                                                          -1,
                                                          -1,
                                                          false,
                                                          atGen,
                                                          nullStateOpt);

    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t q, diopiTensorHandle_t k, diopiTensorHandle_t v,
                                           diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k, int64_t max_q, int64_t max_k,
                                           double dropout_p, bool is_causal, bool return_debug_mask, double scale, diopiTensorHandle_t out,
                                           diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask) {
    impl::aten::setCurStream(ctx);

    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optOut, out);
    auto atCumSeqQ = impl::aten::buildATen(cum_seq_q);
    auto atCumSeqK = impl::aten::buildATen(cum_seq_k);
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);

    auto headSize = atQ.sizes()[3];
    TORCH_CHECK(headSize % 8 == 0, "DIOPI now only support head sizes which are multiple of 8");

    c10::optional<at::Tensor> nullSeqOpt;
    c10::optional<at::Tensor> nullSlopesOpt;
    std::vector<at::Tensor> result = DIOPI_EXT_CALL_FLASH(mha_varlen_fwd,
                                                          atQ,
                                                          atK,
                                                          atV,
                                                          optOut,
                                                          atCumSeqQ,
                                                          atCumSeqK,
                                                          nullSeqOpt,
                                                          nullSlopesOpt,
                                                          max_q,
                                                          max_k,
                                                          dropout_p,
                                                          scale,
                                                          false,
                                                          is_causal,
                                                          -1,
                                                          -1,
                                                          return_debug_mask,
                                                          atGen);

    // PERF: these copy can be eliminated by modifying the flash_attn api
    impl::aten::updateATen2Tensor(ctx, result[fa::mha_fwd_ret_idx::SOFTMAX_LSE], softmax_lse);
    if (return_debug_mask) {
        impl::aten::updateATen2Tensor(ctx, result[fa::mha_fwd_ret_idx::DEBUG_ATTN_MASK], debug_attn_mask);
    }

    return diopiSuccess;
}

diopiError_t diopiMultiHeadAttentionVarLenBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                                   diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                   diopiConstTensorHandle_t softmax_lse, diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k,
                                                   int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, diopiGeneratorHandle_t gen, double scale,
                                                   diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v) {
    impl::aten::setCurStream(ctx);

    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atGen = buildGeneratorForMha(ctx, gen, dropout_p);
    auto atGradOut = impl::aten::buildATen(grad_out);
    auto atOut = impl::aten::buildATen(out);
    auto atLogsumexp = impl::aten::buildATen(softmax_lse);
    auto atCumSeqQ = impl::aten::buildATen(cum_seq_q);
    auto atCumSeqK = impl::aten::buildATen(cum_seq_k);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradQ, grad_q);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradK, grad_k);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradV, grad_v);
    c10::optional<at::Tensor> nullStateOpt;  // Workaround: flash_attn uses non-const optional& as args (which is a really bad idea)
    c10::optional<at::Tensor> nullSlopesOpt;

    std::vector<at::Tensor> result = DIOPI_EXT_CALL_FLASH(mha_varlen_bwd,
                                                          atGradOut,
                                                          atQ,
                                                          atK,
                                                          atV,
                                                          atOut,
                                                          atLogsumexp,
                                                          optGradQ,
                                                          optGradK,
                                                          optGradV,
                                                          atCumSeqQ,
                                                          atCumSeqK,
                                                          nullSlopesOpt,
                                                          max_q,
                                                          max_k,
                                                          dropout_p,
                                                          scale,
                                                          false,
                                                          is_causal,
                                                          -1,
                                                          -1,
                                                          false,
                                                          atGen,
                                                          nullStateOpt);

    return diopiSuccess;
}

diopiError_t diopiFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out, diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen,
                                 diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t alibi_slopes,
                                 float p_dropout, float softmax_scale, bool is_causal, int32_t window_size_left, int32_t window_size_right) {
    impl::aten::setCurStream(ctx);

    // handle param[out]
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optOut, attention_out);

    // handle param[in]
    auto atGen = buildGeneratorForMha(ctx, gen, p_dropout);
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optLinearBiasSlopes, alibi_slopes);

    auto headSize = atQ.sizes()[3];
    DIOPI_CHECK(headSize % 8 == 0, "DIOPI torch impl now only support head sizes which are multiple of 8");

    std::vector<at::Tensor> result = DIOPI_EXT_CALL_FLASH(mha_fwd,
                                                          atQ,
                                                          atK,
                                                          atV,
                                                          optOut,
                                                          optLinearBiasSlopes,
                                                          p_dropout,
                                                          softmax_scale,
                                                          is_causal,
                                                          window_size_left,
                                                          window_size_right,
                                                          /*return_softmax=*/false,
                                                          atGen);

    // PERF: these copy can be eliminated by modifying the flash_attn api
    impl::aten::updateATen2Tensor(ctx, result[fa::mha_fwd_ret_idx::SOFTMAX_LSE], softmax_lse);

    return diopiSuccess;
}

diopiError_t diopiFlashAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v,
                                         diopiConstTensorHandle_t grad_output, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q,
                                         diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t alibi_slopes,
                                         diopiConstTensorHandle_t attention_out, diopiConstTensorHandle_t softmax_lse, float p_dropout, float softmax_scale,
                                         bool is_causal, int32_t window_size_left, int32_t window_size_right) {
    impl::aten::setCurStream(ctx);

    // handle param[out]
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradQ, grad_q);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradK, grad_k);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradV, grad_v);

    // handle param[in]
    auto atGradOut = impl::aten::buildATen(grad_output);
    auto atGen = buildGeneratorForMha(ctx, gen, p_dropout);
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atOut = impl::aten::buildATen(attention_out);
    auto atSoftmaxLse = impl::aten::buildATen(softmax_lse);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optLinearBiasSlopes, alibi_slopes);

    c10::optional<at::Tensor> nullOpt;  // Workaround: flash_attn uses non-const optional& as args (which is a really bad idea)
    std::vector<at::Tensor> result = DIOPI_EXT_CALL_FLASH(mha_bwd,
                                                          atGradOut,
                                                          atQ,
                                                          atK,
                                                          atV,
                                                          atOut,
                                                          atSoftmaxLse,
                                                          optGradQ,
                                                          optGradK,
                                                          optGradV,
                                                          optLinearBiasSlopes,
                                                          p_dropout,
                                                          softmax_scale,
                                                          is_causal,
                                                          window_size_left,
                                                          window_size_right,
                                                          /*deterministic=*/false,
                                                          atGen,
                                                          nullOpt);

    return diopiSuccess;
}

diopiError_t diopiFlashAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out, diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen,
                                       diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t cum_seq_q,
                                       diopiConstTensorHandle_t cum_seq_kv, diopiConstTensorHandle_t alibi_slopes, int32_t max_seqlen_q, int32_t max_seqlen_kv,
                                       float p_dropout, float softmax_scale, bool is_causal, int32_t window_size_left, int32_t window_size_right) {
    impl::aten::setCurStream(ctx);

    // handle param[out]
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optOut, attention_out);

    // handle param[in]
    auto atGen = buildGeneratorForMha(ctx, gen, p_dropout);
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atCumSeqQ = impl::aten::buildATen(cum_seq_q);
    auto atCumSeqKV = impl::aten::buildATen(cum_seq_kv);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optLinearBiasSlopes, alibi_slopes);

    auto headSize = atQ.sizes()[3];
    DIOPI_CHECK(headSize % 8 == 0, "DIOPI torch impl now only support head sizes which are multiple of 8");

    c10::optional<at::Tensor> nullOpt;
    std::vector<at::Tensor> result = DIOPI_EXT_CALL_FLASH(mha_varlen_fwd,
                                                          atQ,
                                                          atK,
                                                          atV,
                                                          optOut,
                                                          atCumSeqQ,
                                                          atCumSeqKV,
                                                          nullOpt,
                                                          optLinearBiasSlopes,
                                                          max_seqlen_q,
                                                          max_seqlen_kv,
                                                          p_dropout,
                                                          softmax_scale,
                                                          /*zero_tensors=*/false,
                                                          is_causal,
                                                          window_size_left,
                                                          window_size_right,
                                                          /*return_softmax=*/false,
                                                          atGen);

    // PERF: these copy can be eliminated by modifying the flash_attn api
    impl::aten::updateATen2Tensor(ctx, result[fa::mha_fwd_ret_idx::SOFTMAX_LSE], softmax_lse);

    return diopiSuccess;
}

diopiError_t diopiFlashAttentionVarLenBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v,
                                               diopiConstTensorHandle_t grad_output, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q,
                                               diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t cum_seq_q,
                                               diopiConstTensorHandle_t cum_seq_kv, diopiConstTensorHandle_t alibi_slopes,
                                               diopiConstTensorHandle_t attention_out, diopiConstTensorHandle_t softmax_lse, int32_t max_seqlen_q,
                                               int32_t max_seqlen_kv, float p_dropout, float softmax_scale, bool is_causal, int32_t window_size_left,
                                               int32_t window_size_right) {
    impl::aten::setCurStream(ctx);

    // handle param[out]
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradQ, grad_q);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradK, grad_k);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optGradV, grad_v);

    // handle param[in]
    auto atGradOut = impl::aten::buildATen(grad_output);
    auto atGen = buildGeneratorForMha(ctx, gen, p_dropout);
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atV = impl::aten::buildATen(v);
    auto atCumSeqQ = impl::aten::buildATen(cum_seq_q);
    auto atCumSeqKV = impl::aten::buildATen(cum_seq_kv);
    auto atOut = impl::aten::buildATen(attention_out);
    auto atSoftmaxLse = impl::aten::buildATen(softmax_lse);
    DIOPI_IMPL_BUILD_ATEN_OPTIONAL(optLinearBiasSlopes, alibi_slopes);

    c10::optional<at::Tensor> nullOpt;  // Workaround: flash_attn uses non-const optional& as args (which is a really bad idea)
    std::vector<at::Tensor> result = DIOPI_EXT_CALL_FLASH(mha_varlen_bwd,
                                                          atGradOut,
                                                          atQ,
                                                          atK,
                                                          atV,
                                                          atOut,
                                                          atSoftmaxLse,
                                                          optGradQ,
                                                          optGradK,
                                                          optGradV,
                                                          atCumSeqQ,
                                                          atCumSeqKV,
                                                          optLinearBiasSlopes,
                                                          max_seqlen_q,
                                                          max_seqlen_kv,
                                                          p_dropout,
                                                          softmax_scale,
                                                          /*zero_tensors=*/false,
                                                          is_causal,
                                                          window_size_left,
                                                          window_size_right,
                                                          /*deterministic=*/false,
                                                          atGen,
                                                          nullOpt);

    return diopiSuccess;
}

}  // namespace impl::cuda
