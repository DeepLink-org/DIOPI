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
    return diopiSuccess;
}

diopiError_t diopiApplyPenalty(
    diopiContextHandle_t ctx,
    diopiTensorHandle_t Logits,
    diopiConstTensorHandle_t presence_penalty,
    diopiConstTensorHandle_t frequency_penalty,
    diopiConstTensorHandle_t p_token_ids,
    diopiConstTensorHandle_t p_token_counts,
    diopiConstTensorHandle_t p_cumsum_seq_len,
    int p_max_len_in_batch
) {
    impl::aten::setCurCtx(ctx);
    auto atLogits = impl::aten::buildATen(Logits);
    auto atPresencePenalty = impl::aten::buildATen(presence_penalty);
    auto atFrequencyPenalty = impl::aten::buildATen(frequency_penalety);
    auto atPTokenIds = impl::aten::buildATen(p_token_ids);
    auto atPTokenCounts = impl::aten::buildATen(p_token_counts);
    auto atPCumsumSeqLen = impl::aten::buildATen(p_cumsum_seq_len);

    // 确保Logits是连续的
    assert(atLogits.is_contiguous());

    // 计算块的大小BLOCK
    int BLOCK = (p_max_len_in_batch > 1024) ? 1024 : p_max_len_in_batch;

    // 计算网格和块的大小
    dim3 grid(atLogits.size(0)); // 线程网格在第0维上有 Logits.size(0) 个线程块
    dim3 block(BLOCK); // 线程块在每个维度上都有BLOCK个线程

    // 在CUDA流上运行_fwd_kernel_apply_penalty函数，并传递相应的参数
    AT_DISPATCH_FLOATING_TYPES(Logits.scalar_type(), "apply_penalty_cuda", ([&] {
        _fwd_kernel_apply_penalty<<<grid, block, 0, stream>>>(
            atLogits,
            atPresencePenalty,
            atFrequencyPenalty,
            atPTokenIds,
            atPTokenCounts,
            atPCumsumSeqLen,
            atLogits.stride(0),
            atLogits.stride(1),
            BLOCK
        );
    }));

    return diopiSuccess;
}


}  // extern "C"
