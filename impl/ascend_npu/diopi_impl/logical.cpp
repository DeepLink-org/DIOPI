/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    op_api::logical_and_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    op_api::logical_or_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
