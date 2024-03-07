/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(out, input, other);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::le_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, other);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::le_(inputAt, otherAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(out, input, other);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::le_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::le_(inputAt, otherAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
