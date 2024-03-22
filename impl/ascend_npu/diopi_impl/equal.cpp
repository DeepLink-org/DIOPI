/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {
diopiError_t diopiEqual(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(out, input, other);
    EXEC_NPU_CMD(aclnnEqual, inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
