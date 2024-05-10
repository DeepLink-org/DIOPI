/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/acloprunner.hpp"
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

void logInternal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, float base) {
    AclOpRunner<1, 1>("Log", ctx).addInput(input).setAttr<float>("base", base).setAttr<float>("scale", 1).setAttr<float>("shift", 0).addOutput(out).run();
}

void logInpInternal(diopiContextHandle_t ctx, diopiTensorHandle_t input, float base) {
    AclOpRunner<1, 1>("Log", ctx).addInput(input).setAttr<float>("base", base).setAttr<float>("scale", 1).setAttr<float>("shift", 0).addOutput(input).run();
}

diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    logInternal(ctx, out, input, -1);
    return diopiSuccess;
}

diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    logInpInternal(ctx, input, -1);
    return diopiSuccess;
}

diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    logInternal(ctx, out, input, 2);
    return diopiSuccess;
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    logInpInternal(ctx, input, 2);
    return diopiSuccess;
}

diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    logInternal(ctx, out, input, 10);
    return diopiSuccess;
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    logInpInternal(ctx, input, 10);
    return diopiSuccess;
}

diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Exp", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiExp(ctx, input, input); }

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Reciprocal", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiReciprocal(ctx, input, input); }

}  // namespace ascend
}  // namespace impl
