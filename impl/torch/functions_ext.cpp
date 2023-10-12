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

diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out1, diopiTensorHandle_t out2, diopiConstTensorHandle_t x1,
                                  diopiConstTensorHandle_t x2, diopiConstTensorHandle_t cos, diopiConstTensorHandle_t sin, const bool conj) {
    impl::aten::setCurCtx(ctx);
    auto atOut1 = impl::aten::buildATen(out1);
    auto atOut2 = impl::aten::buildATen(out2);
    auto atX1 = impl::aten::buildATen(x1);
    auto atX2 = impl::aten::buildATen(x2);
    auto atCos = impl::aten::buildATen(cos);
    auto atSin = impl::aten::buildATen(sin);
    ext::ops::apply_rotary_cuda(atX1, atX2, atCos, atSin, atOut1, atOut2, conj);
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

}  // extern "C"
