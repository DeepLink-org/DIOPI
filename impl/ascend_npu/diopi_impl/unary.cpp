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

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    BEGIN_CALL_ACL_OP(input, value);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::fill_(inputAt, valueAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    if (inputAt.numel() <= 0) {
        return diopiSuccess;
    }

    if (false) {
        acl_op::abs_out(inputAt, outAt);
    } else {
        op_api::abs_out(inputAt, outAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    if (inputAt.numel() <= 0) {
        return diopiSuccess;
    }

    if (false) {
        acl_op::abs_(inputAt);
    } else {
        op_api::abs_(inputAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
