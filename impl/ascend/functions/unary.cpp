/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnNeg, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceNeg, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnRsqrt, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceRsqrt, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSqrt, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceSqrt, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnErf, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceErf, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLog, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLog, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLog2, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLog2, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLog10, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLog10, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnExp, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceExp, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnReciprocal, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceReciprocal, ctx, input);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
