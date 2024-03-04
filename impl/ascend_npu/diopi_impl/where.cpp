/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {
diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                        diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(condition, input, other, out);
    at::Tensor outTemp = op_api::where(conditionAt, inputAt, otherAt);
    outAt.copy_(outTemp);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
