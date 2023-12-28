/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    if (out == nullptr || input == nullptr || !inputAt.defined() || !outAt.defined() || inputAt.numel() <= 0 || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    // TODO(zhaoguochun): This must be repaired
    // outAt.copy_(inputAt);
    outAt.copy_(inputAt.cpu().to(outAt.scalar_type()));
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
