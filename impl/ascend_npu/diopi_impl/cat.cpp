/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    BEGIN_CALL_ACL_OP(out);
    at::Tensor outTempAt = outAt;
    if (outAt.scalar_type() == at::kDouble) {
        outTempAt = outAt.to(at::kFloat);
    } else if (outAt.scalar_type() == at::kLong) {
        outTempAt = outAt.to(at::kInt);
    }

    std::vector<at::Tensor> tensorsAt;
    tensorsAt.reserve(numInputs);
    for (int i = 0; i < numInputs; i++) {
        auto tensorAt = impl::aten::buildATen(tensors[i]);
        if (!tensorAt.defined() || tensorAt.numel() <= 0) {
            continue;
        }
        tensorsAt.push_back(tensorAt.to(outTempAt.scalar_type()));
    }
    if (!tensorsAt.empty()) {
        acl_op::cat_out(tensorsAt, dim, outTempAt);
    }
    if (outAt.scalar_type() != outTempAt.scalar_type()) {
        outAt.copy_(outTempAt);
    }

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
