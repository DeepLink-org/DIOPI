/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <climits>
#include <limits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    dtype = diopi_dtype_float64;

    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, dtype);

    diopiTensorHandle_t minTmp, maxTmp, boolOut;
    makeTensorLike(ctx, &minTmp, input, dtype);
    makeTensorLike(ctx, &maxTmp, input, dtype);
    makeTensorLike(ctx, &boolOut, input, diopi_dtype_bool);

    AscendTensor Tem(input);
    const std::vector<int64_t> sizes = Tem.shape();

    if (min != nullptr) {
        broadcast(ctx, minTmp, min, sizes);
    } else {
        fillTensor(ctx, minTmp, -std::numeric_limits<double>::max());
    }

    if (max != nullptr) {
        broadcast(ctx, maxTmp, max, sizes);
    } else {
        fillTensor(ctx, maxTmp, std::numeric_limits<double>::max());
    }

    // Perform a clamp operation according PyTorch's special handling of the case when max is less than min.
    // In this case, update the value of min to be equal to max to ensure correct behavior.
    diopiLt(ctx, boolOut, maxTmp, minTmp);
    diopiMaskedFill(ctx, minTmp, minTmp, boolOut, maxTmp);

    runner.addInput(minTmp, dtype).addInput(maxTmp, dtype).addOutput(out).run();

    return diopiSuccess;
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    diopiDtype_t dtype;
    dtype = diopi_dtype_float64;

    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, dtype);

    if (min != nullptr && max != nullptr) {
        double minn = getValue<double>(min);
        double maxn = getValue<double>(max);
        if (maxn < minn) {
            runner.addConstInput(*max, dtype).addConstInput(*max, dtype).addOutput(out).run();
            return diopiSuccess;
        }
    }

    if (min != nullptr) {
        runner.addConstInput(*min, dtype);
    } else {
        runner.addConstInput(-std::numeric_limits<double>::max(), dtype);
    }

    if (max != nullptr) {
        runner.addConstInput(*max, dtype);
    } else {
        runner.addConstInput(std::numeric_limits<double>::max(), dtype);
    }

    runner.addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    return diopiClamp(ctx, input, input, min, max);
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    return diopiClampScalar(ctx, input, input, min, max);
}

DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    return diopiClampMinScalar(ctx, input, input, min);
}

DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    return diopiClampMin(ctx, input, input, min);
}

DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    return diopiClampScalar(ctx, out, input, min, nullptr);
}

DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    return diopiClamp(ctx, out, input, min, nullptr);
}

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    return diopiClampMaxScalar(ctx, input, input, max);
}

DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    return diopiClampMax(ctx, input, input, max);
}

DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    return diopiClampScalar(ctx, out, input, nullptr, max);
}

DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    return diopiClamp(ctx, out, input, nullptr, max);
}

}  // namespace ascend
}  // namespace impl
