/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    if (out == input || out == nullptr || input == nullptr || !inputAt.defined() || !outAt.defined() || inputAt.numel() <= 0 || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    auto dtype = outAt.scalar_type();
    EXEC_NPU_CMD(aclnnCast, inputAt, dtype, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
