/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <climits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input);
    if (min != nullptr) {
        runner.addInput(min);
    } else {
        diopiTensorHandle_t minTmp;
        makeTensorLike(ctx, &minTmp, input, dtype);
        if (isIntegralType(dtype)) {
            fillTensor(ctx, &minTmp, static_cast<float>(INT_MIN));
        } else {
            fillTensor(ctx, &minTmp, static_cast<float>(-FLT_MAX));
        }
        runner.addInput(minTmp);
    }
    if (max != nullptr) {
        runner.addInput(max);
    } else {
        diopiTensorHandle_t maxTmp;
        makeTensorLike(ctx, &maxTmp, input, dtype);
        if (isIntegralType(dtype)) {
            fillTensor(ctx, &maxTmp, static_cast<float>(INT_MAX));
        } else {
            fillTensor(ctx, &maxTmp, static_cast<float>(FLT_MAX));
        }
        runner.addInput(maxTmp);
    }
    runner.addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    return diopiClamp(ctx, input, input, min, max);
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input);
    if (min != nullptr) {
        runner.addConstInput(*min, dtype);
    } else {
        if (isIntegralType(dtype)) {
            runner.addConstInput(INT_MIN, dtype);
        } else {
            runner.addConstInput(-FLT_MAX, dtype);
        }
    }
    if (max != nullptr) {
        runner.addConstInput(*max, dtype);
    } else {
        if (isIntegralType(dtype)) {
            runner.addConstInput(INT_MAX, dtype);
        } else {
            runner.addConstInput(FLT_MAX, dtype);
        }
    }
    runner.addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    return diopiClampScalar(ctx, input, input, min, max);
}

}  // namespace ascend
}  // namespace impl
