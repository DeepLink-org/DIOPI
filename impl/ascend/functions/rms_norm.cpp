/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t inv_rms, diopiConstTensorHandle_t input,
                          diopiSize_t normalized_shape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    AclOpRunner<2, 2>("RmsNorm", ctx).addInput(input).addInput(weight).addOutput(out).addOutput(inv_rms).setAttr("epsilon", static_cast<float>(eps)).run();
    return diopiSuccess;
}

diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                  diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias, diopiConstTensorHandle_t inv_rms, diopiSize_t normalized_shape, double eps) {
    AclOpRunner<4, 2>("RmsNorm", ctx)
        .addInput(grad_output)
        .addInput(input)
        .addInput(inv_rms)
        .addInput(weight)
        .addOutput(grad_input)
        .addOutput(grad_weight)
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
