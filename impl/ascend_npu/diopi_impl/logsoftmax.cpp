/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
namespace OP_IMPL_NS {
diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    BEGIN_CALL_ACL_OP(input, out);
    if (inputAt.dim() == 0) {
        op_api::fill_(outAt, 0);
        return diopiSuccess;
    }

    EXEC_NPU_CMD(aclnnLogSoftmax, inputAt, dim, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                     diopiConstTensorHandle_t output, int64_t dim) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, output);
    if (gradInputAt.dim() == 0) {
        op_api::fill_(gradInputAt, 0);
        return diopiSuccess;
    }

    at::ScalarType inputDtype = gradInputAt.scalar_type();
    op_api::_log_softmax_backward_data_out(gradOutputAt, outputAt, dim, inputDtype, gradInputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
