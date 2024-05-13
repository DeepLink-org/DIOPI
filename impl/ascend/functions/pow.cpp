/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnPowTensorTensor, ctx, input, exponent, out);
    return diopiSuccess;
}

diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplacePowTensorTensor, ctx, input, exponent);
    return diopiSuccess;
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnPowTensorScalar, ctx, input, exponent, out);
    return diopiSuccess;
}

diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplacePowTensorScalar, ctx, input, exponent);
    return diopiSuccess;
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnPowScalarTensor, ctx, input, exponent, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
