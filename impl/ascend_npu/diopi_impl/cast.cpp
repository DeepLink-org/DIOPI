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
    if (out == input || out == nullptr || input == nullptr || !inputAt.defined() || !outAt.defined() || inputAt.numel() <= 0 || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    outAt.copy_(inputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
