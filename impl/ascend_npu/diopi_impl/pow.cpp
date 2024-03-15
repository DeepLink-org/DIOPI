/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    BEGIN_CALL_ACL_OP(input, exponent, out);
    // EXEC_NPU_CMD(aclnnPowTensorTensor, inputAt, exponentAt, outAt);
    op_api::pow_out(inputAt, exponentAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    BEGIN_CALL_ACL_OP(input, exponent);
    // EXEC_NPU_CMD(aclnnInplacePowTensorTensor, input, exponent);
    op_api::pow_(inputAt, exponentAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    BEGIN_CALL_ACL_OP(input, exponent, out);
    EXEC_NPU_CMD(aclnnPowTensorScalar, inputAt, exponentAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    BEGIN_CALL_ACL_OP(input, exponent);
    EXEC_NPU_CMD(aclnnInplacePowTensorScalar, inputAt, exponentAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    BEGIN_CALL_ACL_OP(input, exponent, out);
    EXEC_NPU_CMD(aclnnPowScalarTensor, inputAt, exponentAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
