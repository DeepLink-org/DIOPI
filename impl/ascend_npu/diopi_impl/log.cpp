/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::log_(inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::log_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::log2_(inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::log2_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::log10_(inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::log10_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
