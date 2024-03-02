/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/OpInterface.h"

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

diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    acl_op::sigmoid_out(inputAt, outAt);
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

diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(out, input);
    if (false) {
        acl_op::silu_out(inputAt, outAt);
    } else {
        op_api::silu_out(inputAt, outAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    if (false) {
        acl_op::silu_out(inputAt, inputAt);
    } else {
        op_api::silu_out(inputAt, inputAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, input);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(gradInputAt);
    if (false) {
        acl_op::silu_backward_out(gradOutputAt, inputAt, gradInputAt);
    } else {
        op_api::silu_backward_out(gradOutputAt, inputAt, gradInputAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
