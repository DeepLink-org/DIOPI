/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, out, alpha, other);
    if (!outAt.defined() || outAt.numel() <= 0) {
        return diopiSuccess;
    }

    op_api::add_out(inputAt, otherAt, alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha);
    if (!inputAt.defined() || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::add_(inputAt, otherAt, alphaAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha, out);
    if (!outAt.defined() || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::add_out(inputAt, at::scalar_to_tensor(otherAt), alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha);
    if (!inputAt.defined() || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::add_(inputAt, otherAt, alphaAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
