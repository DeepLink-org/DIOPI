/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    BEGIN_CALL_ACL_OP(out, input, index);
    if (!inputAt.defined() || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    at::Tensor inputTmpAt = inputAt;
    if (inputAt.scalar_type() != outAt.scalar_type()) {
        inputTmpAt = inputAt.to(outAt.scalar_type());
    }
    if (false) {
        acl_op::gather_out(inputTmpAt, dim, indexAt, false, outAt);
    } else {
        op_api::gather_out(inputTmpAt, dim, indexAt, false, outAt);
    }
    END_CALL_ACL_OP();
}

// diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t
// input,
//                                  int64_t dim, diopiConstTensorHandle_t index) {
//
//     BEGIN_CALL_ACL_OP(gradInput, gradOutput, input, index);
//     if (gradOutputAt.numel() <= 0 || !gradOutputAt.defined()) {
//         return diopiSuccess;
//     }
//     // check index
//     TORCH_CHECK((gradOutputAt.dim() == gradInputAt.dim() && gradInputAt.dim() == indexAt.dim()) || indexAt.dim() == 0,
//                 "gradInput,gradOutput,index must have same ndim! only exception is index is empty");
//     if (indexAt.dim() == 0) {
//         gradInputAt.copy_(gradOutputAt);
//         return diopiSuccess;
//     }
//     at::Scalar zero{0.0};
//     op_api::fill_(gradInputAt, zero);
//     // input to output type
//     at::Tensor gradOutputTmpAt = gradOutputAt;
//     if (gradOutputAt.scalar_type() != gradInputAt.scalar_type()) {
//         gradOutputTmpAt = gradOutputAt.to(gradInputAt.scalar_type());
//     }
//     // scatter add
//     bool isSameShape = true;
//     auto gradInputShape = gradInputAt.sizes();
//     auto gradOutputShape = gradOutputAt.sizes();
//     for (int64_t i = 0; i < gradInputAt.dim(); ++i) {
//         if (gradInputShape[i] != gradOutputShape[i]) {
//             isSameShape = false;
//             break;
//         }
//     }
//     if (isSameShape) {
//         int64_t reduction = 1;
//         EXEC_NPU_CMD(aclnnInplaceScatter, gradInputAt, dim, indexAt, gradOutputTmpAt, reduction);
//     } else {
//         acl_op::scatter_add_(gradInputAt, dim, indexAt, gradOutputTmpAt);
//     }
//     END_CALL_ACL_OP();
// }

}  // namespace OP_IMPL_NS
