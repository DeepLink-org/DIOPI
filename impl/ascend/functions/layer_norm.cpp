/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalizedShape,
                            double eps) {
    // gen begin normalized dim
    diopiSize_t inShape;
    diopiGetTensorShape(input, &inShape);
    const int axis = inShape.len - normalizedShape.len;
    int64_t beginDim = axis;

    // call acl op
    AclOpRunner<3, 3>("LayerNorm", ctx)
        .addInput(input)
        .addInput(weight)
        .addInput(bias)
        .addOutput(out)
        .addOutput(saveMean)
        .addOutput(saveInvstd)
        .setAttr("begin_norm_axis", beginDim)
        .setAttr("begin_params_axis", beginDim)
        .setAttr<float>("epsilon", eps)
        .run();
    return diopiSuccess;
}

diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalizedShape) {
    AclOpRunner<5, 3>("LayerNormGrad", ctx)
        .addInput(gradOutput)
        .addInput(input)
        .addInput(rstd)
        .addInput(mean)
        .addInput(weight)
        .addOutput(gradInput)
        .addOutput(gradWeight)
        .addOutput(gradBias)
        .run();
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
