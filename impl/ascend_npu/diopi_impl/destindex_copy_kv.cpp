/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiDestIndexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
    BEGIN_CALL_ACL_OP(out, k, destLoc);
    // handle undefined tensor and empty tensor
    if (!outAt.defined() || outAt.numel() == 0) {
        return diopiSuccess;
    }

    acl_op::_index_put_impl_(outAt, {destLocAt}, kAt, false, false);
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
