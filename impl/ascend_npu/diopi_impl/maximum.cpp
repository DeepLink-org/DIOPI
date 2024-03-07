/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/OpInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other, out);
    op_api::maximum_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t maxIndices, diopiConstTensorHandle_t input, int64_t dim) {
    BEGIN_CALL_ACL_OP(input, maxIndices, max);
    op_api::max_out(inputAt, dim, false, maxAt, maxIndicesAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
