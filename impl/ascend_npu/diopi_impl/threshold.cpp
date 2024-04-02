/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                            const diopiScalar_t* value) {
    BEGIN_CALL_ACL_OP(input, threshold, value, out);
    EXEC_NPU_CMD(aclnnThreshold, inputAt, thresholdAt, valueAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    BEGIN_CALL_ACL_OP(input, threshold, value);
    EXEC_NPU_CMD(aclnnInplaceThreshold, inputAt, thresholdAt, valueAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, input, threshold);
    EXEC_NPU_CMD(aclnnThresholdBackward, gradOutputAt, inputAt, thresholdAt, gradInputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
