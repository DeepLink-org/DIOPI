/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    at::IntArrayRef outSize = outAt.sizes();
    EXEC_NPU_CMD(aclnnExpand, inputAt, outSize, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
