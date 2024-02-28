/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    BEGIN_CALL_ACL_OP(input, out);
    at::Tensor result;
    if (inputAt.scalar_type() == at::kHalf) {
        result = op_api::_softmax(inputAt, dim, true);
    } else {
        result = op_api::_softmax(inputAt, dim, false);
    }
    outAt.copy_(result);
    END_CALL_ACL_OP();
}

diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t output,
                                  int64_t dim) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, output);
    at::ScalarType inputDtype = gradInputAt.scalar_type();
    op_api::_softmax_backward_data_out(gradOutputAt, outputAt, dim, inputDtype, gradInputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    BEGIN_CALL_ACL_OP(input, out);
    at::Tensor result;
    if (inputAt.scalar_type() == at::kHalf) {
        result = op_api::_log_softmax(inputAt, dim, true);
    } else {
        result = op_api::_log_softmax(inputAt, dim, false);
    }
    outAt.copy_(result);
    END_CALL_ACL_OP();
}

diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                     diopiConstTensorHandle_t output, int64_t dim) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, output);
    at::ScalarType inputDtype = gradInputAt.scalar_type();
    op_api::_log_softmax_backward_data_out(gradOutputAt, outputAt, dim, inputDtype, gradInputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
