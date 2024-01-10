/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    auto result = acl_op::relu(inputAt);
    outAt.copy_(result);
    END_CALL_ACL_OP();
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    acl_op::relu_(inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    acl_op::hardswish_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    acl_op::hardswish_(inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, input);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(gradInputAt);
    acl_op::hardswish_backward(gradOutputAt, inputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
