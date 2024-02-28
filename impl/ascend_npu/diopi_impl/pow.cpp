/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    BEGIN_CALL_ACL_OP(input, exponent, out);
    op_api::pow_out(inputAt, exponentAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    BEGIN_CALL_ACL_OP(input, exponent);
    op_api::pow_(inputAt, exponentAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    BEGIN_CALL_ACL_OP(input, exponent, out);
    op_api::pow_out(inputAt, exponentAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    BEGIN_CALL_ACL_OP(input, exponent);
    op_api::pow_(inputAt, exponentAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    BEGIN_CALL_ACL_OP(input, exponent, out);
    op_api::pow_out(inputAt, exponentAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
