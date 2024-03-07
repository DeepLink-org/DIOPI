/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {
diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    BEGIN_CALL_ACL_OP(input, mask);
    at::Tensor outAt = op_api::masked_select(inputAt, maskAt);
    impl::aten::buildDiopiTensor(ctx, outAt, out);
    END_CALL_ACL_OP();
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    BEGIN_CALL_ACL_OP(input, mask, gradInput, gradOutput);
    if (inputAt.dim() < maskAt.dim()) {
        at::Tensor gradInputNew = op_api::zeros_like(maskAt, gradInputAt.scalar_type(), maskAt.layout(), maskAt.device(), false);
        op_api::masked_scatter_(gradInputNew, maskAt, gradOutputAt);
        std::vector<int64_t> dims;
        dims.reserve(32);
        for (int i = 0; i < maskAt.dim() - inputAt.dim(); i++) {
            dims.push_back(i);
        }
        at::OptionalIntArrayRef sumDim(dims);
        op_api::sum_out(gradInputNew, sumDim, false, gradInputAt.scalar_type(), gradInputAt);
    } else {
        at::Scalar zero = 0;
        op_api::fill_(gradInputAt, zero);
        op_api::masked_scatter_(gradInputAt, maskAt, gradOutputAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
