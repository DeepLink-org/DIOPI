/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    BEGIN_CALL_ACL_OP(input);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    BEGIN_CALL_ACL_OP(out);
    op_api::triu_out(inputAt, diagonal, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
    BEGIN_CALL_ACL_OP(input);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::triu_(inputAt, diagonal);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
