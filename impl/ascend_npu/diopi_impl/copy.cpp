/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
void calStride(diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat, diopiSize_t* stride) {}

namespace OP_IMPL_NS {

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    BEGIN_CALL_ACL_OP(src, dest);
    if (src == nullptr || dest == nullptr || !srcAt.defined() || !destAt.defined() || srcAt.numel() <= 0 || destAt.numel() <= 0) {
        return diopiSuccess;
    }
    at_npu::native::NPUNativeOpApiFunctions::copy_(destAt, srcAt, true);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
