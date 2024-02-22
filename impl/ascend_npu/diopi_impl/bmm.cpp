/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    BEGIN_CALL_ACL_OP(out);
    if (outAt.numel() == 0) {
        return diopiSuccess;
    }
    BEGIN_CALL_ACL_OP(input, mat2);
    op_api::bmm_out(inputAt, mat2At, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
