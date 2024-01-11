/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    at::Tensor tensor0 = impl::aten::buildATen(tensors[0]);
    if (tensor0.numel() == 0) {
        return diopiSuccess;
    }

    BEGIN_CALL_ACL_OP(out);
    std::vector<at::Tensor> tensorsVec(numTensors);
    for (size_t i = 0; i < numTensors; i++) {
        tensorsVec[i] = impl::aten::buildATen(tensors[i]);
    }
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    acl_op::stack_out(tensorsVec, dim, outAt);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
