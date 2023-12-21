/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRms, diopiConstTensorHandle_t input,
                          diopiSize_t normalizedShape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    AscendTensor inputTensor(input);
    ASCEND_CHECK_ABORT(1 == normalizedShape.len && normalizedShape.data[0] == inputTensor.shape()[inputTensor.dim() - 1], "normalized shape error!");
    AclOpRunner<2, 2>("RmsNorm", ctx).addInput(input).addInput(weight).setAttr("epsilon", static_cast<float>(eps)).addOutput(out).addOutput(invRms).run();
    return diopiSuccess;
}

diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                  diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRms, diopiSize_t normalizedShape, double eps) {
    AscendTensor inputTensor(input);
    ASCEND_CHECK_ABORT(1 == normalizedShape.len && normalizedShape.data[0] == inputTensor.shape()[inputTensor.dim() - 1], "normalized shape error!");
    AclOpRunner<4, 2>("RmsNorm", ctx).addInput(gradOutput).addInput(input).addInput(invRms).addInput(weight).addOutput(gradInput).addOutput(gradWeight).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
