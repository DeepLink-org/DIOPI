/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {
diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                        diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(condition, input, other, out);
    op_api::where_out(conditionAt, inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
