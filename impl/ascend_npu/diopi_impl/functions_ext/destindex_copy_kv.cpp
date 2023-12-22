/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiDestIndexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
    BEGIN_CALL_ACL_OP(out, k, destLoc);
    at::index_put_(outAt, {destLocAt}, kAt, false);
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
