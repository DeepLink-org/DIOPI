/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                                      const diopiScalar_t* value) {
    diopiTensorHandle_t thresholdTensor;
    diopiTensorHandle_t valueTensor;
    makeTensorFromScalar(ctx, threshold, &thresholdTensor);
    makeTensorFromScalar(ctx, value, &valueTensor);
    AclOpRunner<3, 1>("ThresholdV2", ctx).addInput(input).addConstInput(thresholdTensor).addConstInput(valueTensor).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    return diopiThreshold(ctx, input, input, threshold, value);
}

DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    AclOpRunner<2, 1>("ThresholdGradV2D", ctx).addInput(gradOutput, input).setAttr("threshold", getValue<float>(threshold)).addOutput(gradInput).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
