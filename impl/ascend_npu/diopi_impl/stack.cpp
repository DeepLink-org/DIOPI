/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim)
 {
    BEGIN_CALL_ACL_OP(out);
    std::vector<at::Tensor> AtTensors;
    for (int i=0; i < numTensors; i++) {
        auto tensorI = tensors[i];
        AtTensors[i] = BEGIN_CALL_ACL_OP(tensorI);
    }
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    acl_op::stack_out(AtTensors, dim, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
