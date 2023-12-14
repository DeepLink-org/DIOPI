/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    BEGIN_CALL_ACL_OP(src, dest);
    if (!srcAt.defined() || !destAt.defined()) {
        return diopiSuccess;
    }
    at_npu::native::NPUNativeFunctions::copy_(destAt, srcAt, false);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
