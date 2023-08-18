/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t negativeInputRtnFillNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    // get nan value tensor
    diopiTensorHandle_t nanValue;
    auto nanValueScalar = diopiScalar_t();
    nanValueScalar.stype = diopi_dtype_float64;
    nanValueScalar.fval = 0.0;
    makeTensorFromScalar(ctx, &nanValueScalar, &nanValue, diopi_dtype_float32, diopi_device);
    auto zeroValueScalar = diopiScalar_t();
    zeroValueScalar.stype = diopi_dtype_float64;
    zeroValueScalar.fval = 0.0;
    diopiDivInpScalar(ctx, nanValue, &zeroValueScalar, diopiRoundMode_t::RoundModeNone);

    // get negative mask
    diopiTensorHandle_t mask;
    makeTensorLike(ctx, &mask, input, diopi_dtype_bool);
    diopiLtScalar(ctx, mask, input, &zeroValueScalar);

    // masked_fill nan
    return diopiMaskedFillInp(ctx, out, mask, nanValue);
}

extern "C" {
DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Neg", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiNeg(ctx, input, input); }

DIOPI_API diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Rsqrt", ctx).addInput(input, ACL_FORMAT_ND).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiRsqrt(ctx, input, input); }

DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Sqrt", ctx).addInput(input).addOutput(out).run();
    negativeInputRtnFillNan(ctx, out, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Log", ctx).addInput(input).setAttr<float>("base", -1.0).setAttr<float>("scale", 1.0).setAttr<float>("shift", 0.0).addOutput(out).run();
    negativeInputRtnFillNan(ctx, out, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Floor", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
