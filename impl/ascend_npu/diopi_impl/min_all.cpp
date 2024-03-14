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

at::Tensor minAll(const at::Tensor& self, at::Tensor& result) {
    DO_COMPATIBILITY(aclnnMin, acl_op::min(self));
    EXEC_NPU_CMD(aclnnMin, self, result);
    return result;
}

}  // namespace

diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, min);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    minAll(inputAt, minAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
