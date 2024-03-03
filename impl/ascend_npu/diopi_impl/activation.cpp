/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    EXEC_NPU_CMD(aclnnRelu, inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    EXEC_NPU_CMD(aclnnInplaceRelu, inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    op_api::sigmoid_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    op_api::sigmoid_(inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                  diopiConstTensorHandle_t output) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, output);
    op_api::sigmoid_backward_out(gradOutputAt, outputAt, gradInputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    op_api::hardswish_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    op_api::hardswish_(inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, input);
    EXEC_NPU_CMD(aclnnHardswishBackward, gradOutputAt, inputAt, gradInputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(out, input);
    op_api::silu_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    op_api::silu_(inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, input);
    op_api::silu_backward_out(gradOutputAt, inputAt, gradInputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
