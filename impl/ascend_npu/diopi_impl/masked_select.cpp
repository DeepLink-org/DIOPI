/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {
diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    BEGIN_CALL_ACL_OP(input, mask);
    at::Tensor outAt = acl_op::masked_select(inputAt, maskAt);
    impl::aten::buildDiopiTensor(ctx, outAt, out);
    END_CALL_ACL_OP();
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    BEGIN_CALL_ACL_OP(input, mask, gradInput, gradOutput);
    at::Scalar zero = 0;
    acl_op::fill_(gradInputAt, zero);
    acl_op::masked_scatter_(gradInputAt, maskAt, gradOutputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
