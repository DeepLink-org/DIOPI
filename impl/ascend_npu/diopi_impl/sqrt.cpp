/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/OpInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    if (inputAt.numel() <= 0) {
        return diopiSuccess;
    }

    if (outAt.scalar_type() == at::kComplexFloat || outAt.scalar_type() == at::kComplexDouble) {
        acl_op::sqrt_out(inputAt, outAt);
    } else {
        op_api::sqrt_out(inputAt, outAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    if (inputAt.numel() <= 0) {
        return diopiSuccess;
    }

    if (inputAt.scalar_type() == at::kComplexFloat || inputAt.scalar_type() == at::kComplexDouble) {
        acl_op::sqrt_(inputAt);
    } else {
        op_api::sqrt_(inputAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
