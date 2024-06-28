/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseNot, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseNot, ctx, input, input);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseAndTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseAndTensor, ctx, input, other, input);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseAndScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseAndScalar, ctx, input, other, input);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseOrTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseOrTensor, ctx, input, other, input);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseOrScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBitwiseOrScalar, ctx, input, other, input);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
