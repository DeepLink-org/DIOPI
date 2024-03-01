/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */
#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t outputSize) {
    if (outputSize.data[0] == 0) {
        return diopiSuccess;
    }
    BEGIN_CALL_ACL_OP(input, outputSize, out);
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool2d, inputAt, outputSizeAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, gradOutput, gradInput);
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool2dBackward, gradOutputAt, inputAt, gradInputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
