/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

DIOPI_API diopiError_t diopiIsNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    at::Tensor outTemp = op_api::isclose(inputAt, inputAt);
    op_api::logical_not_out(outTemp, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
