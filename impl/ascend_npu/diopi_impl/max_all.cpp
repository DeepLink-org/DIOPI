/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

namespace {

at::Tensor maxAll(const at::Tensor& self, at::Tensor& result) {
    DO_COMPATIBILITY(aclnnMax, acl_op::max(self));
    EXEC_NPU_CMD(aclnnMax, self, result);
    return result;
}

}  // namespace

diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, max);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    maxAll(inputAt, maxAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
