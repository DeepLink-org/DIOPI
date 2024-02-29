/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {
diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    op_api::bitwise_not_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    op_api::bitwise_not_out(inputAt, inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other, out);
    op_api::bitwise_and_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other);
    op_api::bitwise_and_out(inputAt, otherAt, inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, other, out);
    op_api::bitwise_and_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, other);
    op_api::bitwise_and_out(inputAt, otherAt, inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other, out);
    op_api::bitwise_or_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other);
    op_api::bitwise_or_out(inputAt, otherAt, inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, other, out);
    op_api::bitwise_or_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, other);
    op_api::bitwise_or_out(inputAt, otherAt, inputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
