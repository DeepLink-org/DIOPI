/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    BEGIN_CALL_ACL_OP(input, mat2, out);
    acl_op::bmm_out(inputAt, mat2At, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
