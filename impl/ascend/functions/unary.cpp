/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Neg", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiNeg(ctx, input, input); }

diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Rsqrt", ctx).addInput(input, ACL_FORMAT_ND).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiRsqrt(ctx, input, input); }

diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Sqrt", ctx).addInput(input).addOutput(out).run();
    negativeInputRtnFillNan(ctx, out, input);
    return diopiSuccess;
}

diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiSqrt(ctx, input, input); }

diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Erfinv", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiErf(ctx, input, input); }

DIOPI_API diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiDtype_t inputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    if (inputDtype == diopi_dtype_uint8 || inputDtype == diopi_dtype_bool) {
        AscendTensor inCopy(input);
        castTensor(ctx, inCopy, diopi_dtype_int16);
        AclOpRunner<1, 1>("Abs", ctx).addInput(inCopy).addOutput(out).run();
    } else {
        AclOpRunner<1, 1>("Abs", ctx).addInput(input).addOutput(out).run();
    }
    return diopiSuccess;
}

diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiAbs(ctx, input, input); }

void logInternal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, float base) {
    AclOpRunner<1, 1>("Log", ctx).addInput(input).setAttr<float>("base", base).setAttr<float>("scale", 1).setAttr<float>("shift", 0).addOutput(out).run();
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    if (diopi_dtype_float64 != dtype) {
        negativeInputRtnFillNan(ctx, out, input);
    }
}

void logInpInternal(diopiContextHandle_t ctx, diopiTensorHandle_t input, float base) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    if (diopi_dtype_float64 != dtype) {
        negativeInputRtnFillNan(ctx, input, input);
    }
    negativeInputRtnFillNan(ctx, input, input);
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
