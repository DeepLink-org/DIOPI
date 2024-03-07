/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(out, input, other);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::ne_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, other);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::ne_(inputAt, otherAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(out, input, other);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::ne_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::ne_(inputAt, otherAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
