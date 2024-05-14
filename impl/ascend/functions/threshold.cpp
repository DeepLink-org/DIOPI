/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                            const diopiScalar_t* value) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnThreshold, ctx, input, threshold, value, out);
    return diopiSuccess;
}

diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceThreshold, ctx, input, threshold, value);
    return diopiSuccess;
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnThresholdBackward, ctx, gradOutput, input, threshold, gradInput);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
