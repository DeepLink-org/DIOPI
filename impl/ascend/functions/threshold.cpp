/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                            const diopiScalar_t* value) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    float inputThreshold = getValue<float>(threshold);
    float inputValue = getValue<float>(value);

    if (dtype == diopi_dtype_uint8) {
        // Converting a float type directly to uint is undefined behavior, so it is necessary to first convert it to an int type.
        inputThreshold = static_cast<float>(static_cast<uint8_t>(static_cast<int>(inputThreshold)));
        inputValue = static_cast<float>(static_cast<uint8_t>(static_cast<int>(inputValue)));
    } else if (isIntegralType(dtype)) {
        inputThreshold = static_cast<float>(getValue<int>(threshold));
        inputValue = static_cast<float>(getValue<int>(value));
    }
    AclOpRunner<1, 1>("ThresholdV2D", ctx).addInput(input).setAttr("threshold", inputThreshold).setAttr("value", inputValue).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    return diopiThreshold(ctx, input, input, threshold, value);
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    AclOpRunner<2, 1>("ThresholdGradV2D", ctx).addInput(gradOutput).addInput(input).setAttr("threshold", getValue<float>(threshold)).addOutput(gradInput).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
