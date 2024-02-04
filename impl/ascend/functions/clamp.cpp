/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <climits>
#include <limits>
#include <map>
#include <string>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

std::pair<double, double> getMinMaxFromDtype(diopiDtype_t inputDtype) {
    switch (inputDtype) {
        case diopi_dtype_int8:
            return std::make_pair(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
        case diopi_dtype_uint8:
            return std::make_pair(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
        case diopi_dtype_int16:
            return std::make_pair(std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max());
        case diopi_dtype_uint16:
            return std::make_pair(std::numeric_limits<uint16_t>::min(), std::numeric_limits<uint16_t>::max());
        case diopi_dtype_int32:
            return std::make_pair(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
        case diopi_dtype_uint32:
            return std::make_pair(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
        case diopi_dtype_int64:
            return std::make_pair(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
        case diopi_dtype_uint64:
            return std::make_pair(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());
        case diopi_dtype_float16:
            return std::make_pair(-std::numeric_limits<half_float::half>::max(), std::numeric_limits<half_float::half>::max());
        case diopi_dtype_float32:
            return std::make_pair(-std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        case diopi_dtype_float64:
            return std::make_pair(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        case diopi_dtype_bool:
            return std::make_pair(std::numeric_limits<bool>::min(), std::numeric_limits<bool>::max());
        default:
            break;
    }
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);

    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, dtype);

    diopiTensorHandle_t minTmp, maxTmp, boolOut;
    makeTensorLike(ctx, &minTmp, input, dtype);
    makeTensorLike(ctx, &maxTmp, input, dtype);
    makeTensorLike(ctx, &boolOut, input, diopi_dtype_bool);

    AscendTensor temp(input);
    const std::vector<int64_t>& sizes = temp.shape();

    if (min != nullptr) {
        broadcast(ctx, minTmp, min, sizes);
    } else {
        auto min_val = getMinMaxFromDtype(dtype).first;
        fillTensor(ctx, minTmp, min_val);
    }

    if (max != nullptr) {
        broadcast(ctx, maxTmp, max, sizes);
    } else {
        auto max_val = getMinMaxFromDtype(dtype).second;
        fillTensor(ctx, maxTmp, max_val);
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
    diopiGetTensorDtype(input, &dtype);
    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input);

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
        auto min_val = getMinMaxFromDtype(dtype).first;
        diopiScalar_t min_val_scalar = constructDiopiScalarT(diopi_dtype_float64, min_val);
        runner.addConstInput(min_val_scalar, dtype);
    }

    if (max != nullptr) {
        runner.addConstInput(*max, dtype);
    } else {
        auto max_val = getMinMaxFromDtype(dtype).second;
        diopiScalar_t max_val_scalar = constructDiopiScalarT(diopi_dtype_float64, max_val);
        runner.addConstInput(max_val_scalar, dtype);
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
