/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    BEGIN_CALL_ACL_OP(input, dims, out);
    EXEC_NPU_CMD(aclnnFlip, inputAt, dimsAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
