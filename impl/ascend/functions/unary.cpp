/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Neg", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiNeg(ctx, input, input); }

extern "C" DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
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
