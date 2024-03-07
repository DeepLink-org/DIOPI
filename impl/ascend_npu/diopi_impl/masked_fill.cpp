/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                             diopiConstTensorHandle_t value) {
    BEGIN_CALL_ACL_OP(out, input, mask, value);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    if (outAt.data_ptr() != inputAt.data_ptr()) {
        outAt.copy_(inputAt);
    }
    op_api::masked_fill_(outAt, maskAt, valueAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    BEGIN_CALL_ACL_OP(input, mask, value);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::masked_fill_(inputAt, maskAt, valueAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                   const diopiScalar_t* value) {
    BEGIN_CALL_ACL_OP(out, input, mask, value);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    if (outAt.data_ptr() != inputAt.data_ptr()) {
        outAt.copy_(inputAt);
    }
    op_api::masked_fill_(outAt, maskAt, valueAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    BEGIN_CALL_ACL_OP(input, mask, value);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::masked_fill_(inputAt, maskAt, valueAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
