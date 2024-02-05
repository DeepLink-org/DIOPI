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

// to get the limit value according to diopiDtype
std::pair<double, double> getMinMaxFromDtype(diopiDtype_t tensorDtype) {
    switch (tensorDtype) {
        case diopi_dtype_int8:
            return std::make_pair(std::numeric_limits<int8_t>::lowest(), std::numeric_limits<int8_t>::max());
        case diopi_dtype_uint8:
            return std::make_pair(std::numeric_limits<uint8_t>::lowest(), std::numeric_limits<uint8_t>::max());
        case diopi_dtype_int16:
            return std::make_pair(std::numeric_limits<int16_t>::lowest(), std::numeric_limits<int16_t>::max());
        case diopi_dtype_uint16:
            return std::make_pair(std::numeric_limits<uint16_t>::lowest(), std::numeric_limits<uint16_t>::max());
        case diopi_dtype_int32:
            return std::make_pair(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
        case diopi_dtype_uint32:
            return std::make_pair(std::numeric_limits<uint32_t>::lowest(), std::numeric_limits<uint32_t>::max());
        case diopi_dtype_int64:
            return std::make_pair(std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max());
        case diopi_dtype_uint64:
            return std::make_pair(std::numeric_limits<uint64_t>::lowest(), std::numeric_limits<uint64_t>::max());
        case diopi_dtype_float16:
            return std::make_pair(std::numeric_limits<half_float::half>::lowest(), std::numeric_limits<half_float::half>::max());
        case diopi_dtype_float32:
            return std::make_pair(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
        case diopi_dtype_float64:
            return std::make_pair(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
        case diopi_dtype_bool:
            return std::make_pair(std::numeric_limits<bool>::lowest(), std::numeric_limits<bool>::max());
        default:
            break;
    }
}

std::pair<double, double> getFloatMinMaxFromDtype(diopiDtype_t tensorDtype) {
    switch (tensorDtype) {
        case diopi_dtype_float16:
            return std::make_pair(std::numeric_limits<half_float::half>::lowest(), std::numeric_limits<half_float::half>::max());
        case diopi_dtype_float32:
            return std::make_pair(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
        case diopi_dtype_float64:
            return std::make_pair(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
        default:
            break;
    }
}

std::pair<int64_t, int64_t> getIntMinMaxFromDtype(diopiDtype_t tensorDtype) {
    switch (tensorDtype) {
        case diopi_dtype_int8:
            return std::make_pair(std::numeric_limits<int8_t>::lowest(), std::numeric_limits<int8_t>::max());
        case diopi_dtype_uint8:
            return std::make_pair(std::numeric_limits<uint8_t>::lowest(), std::numeric_limits<uint8_t>::max());
        case diopi_dtype_int16:
            return std::make_pair(std::numeric_limits<int16_t>::lowest(), std::numeric_limits<int16_t>::max());
        case diopi_dtype_uint16:
            return std::make_pair(std::numeric_limits<uint16_t>::lowest(), std::numeric_limits<uint16_t>::max());
        case diopi_dtype_int32:
            return std::make_pair(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
        case diopi_dtype_uint32:
            return std::make_pair(std::numeric_limits<uint32_t>::lowest(), std::numeric_limits<uint32_t>::max());
        case diopi_dtype_int64:
            return std::make_pair(std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max());
        case diopi_dtype_uint64:
            return std::make_pair(std::numeric_limits<uint64_t>::lowest(), std::numeric_limits<uint64_t>::max());
        case diopi_dtype_bool:
            return std::make_pair(std::numeric_limits<bool>::lowest(), std::numeric_limits<bool>::max());
        default:
            break;
    }
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    diopiDtype_t highDtype, inputDtype, minDtype, maxDtype;
    diopiTensorHandle_t minTmp, maxTmp, boolOut;

    AscendTensor inputAt(input);
    const std::vector<int64_t>& sizes = inputAt.shape();
    inputDtype = inputAt.dtype();

    if (min != nullptr) {
        diopiGetTensorDtype(min, &minDtype);
        makeTensorLike(ctx, &minTmp, input, minDtype);
        broadcast(ctx, minTmp, min, sizes);
    } else {
        makeTensorLike(ctx, &minTmp, input, inputDtype);
        double minVal = getMinMaxFromDtype(inputDtype).first;
        fillTensor(ctx, minTmp, minVal);
        minDtype = inputDtype;
    }

    if (max != nullptr) {
        diopiGetTensorDtype(max, &maxDtype);
        makeTensorLike(ctx, &maxTmp, input, maxDtype);
        broadcast(ctx, maxTmp, max, sizes);
    } else {
        makeTensorLike(ctx, &maxTmp, input, inputDtype);
        double maxVal = getMinMaxFromDtype(inputDtype).second;
        fillTensor(ctx, maxTmp, maxVal);
        maxDtype = inputDtype;
    }

    highDtype = promoteTypes(minDtype, maxDtype);
    highDtype = promoteTypes(inputDtype, highDtype);

    // Perform a clamp operation according PyTorch's special handling of the case when max is less than min.
    // In this case, update the value of min to be equal to max to ensure correct behavior.
    makeTensorLike(ctx, &boolOut, input, diopi_dtype_bool);
    diopiLt(ctx, boolOut, maxTmp, minTmp);
    diopiMaskedFill(ctx, minTmp, minTmp, boolOut, maxTmp);

    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, highDtype).addInput(minTmp, highDtype).addInput(maxTmp, highDtype).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* minPtr,
                              const diopiScalar_t* maxPtr) {
    AscendTensor inputAt(input);
    diopiDtype_t inputDtype, minDtype, maxDtype, highDtype;
    diopiGetTensorDtype(input, &inputDtype);
    diopiScalar_t min, max;
    double minVal, maxVal;

    if (minPtr != nullptr) {
        minDtype = minPtr->stype;
        min = *minPtr;
        if (isFloatingType(min.stype)) {
            minVal = min.fval;
        } else {
            minVal = min.ival;
        }
    } else {
        minDtype = inputDtype;
        if (isFloatingType(inputDtype)) {
            double minLimitVal = getFloatMinMaxFromDtype(inputDtype).first;
            min = constructDiopiScalarT(inputDtype, minLimitVal);
            minVal = minLimitVal;
        } else {
            int64_t minLimitVal = getIntMinMaxFromDtype(inputDtype).first;
            min = constructDiopiScalarT(inputDtype, minLimitVal);
            minVal = minLimitVal;
        }
    }

    if (maxPtr != nullptr) {
        maxDtype = maxPtr->stype;
        max = *maxPtr;
        if (isFloatingType(max.stype)) {
            maxVal = max.fval;
        } else {
            maxVal = max.ival;
        }
    } else {
        maxDtype = inputDtype;
        if (isFloatingType(inputDtype)) {
            double maxLimitVal = getFloatMinMaxFromDtype(inputDtype).second;
            max = constructDiopiScalarT(inputDtype, maxLimitVal);
            maxVal = maxLimitVal;
        } else {
            int64_t maxLimitVal = getIntMinMaxFromDtype(inputDtype).second;
            max = constructDiopiScalarT(inputDtype, maxLimitVal);
            maxVal = maxLimitVal;
        }
    }

    highDtype = promoteTypes(minDtype, maxDtype);
    highDtype = promoteTypes(inputDtype, highDtype);

    // Perform a clamp operation according PyTorch's special handling of the case when max is less than min.
    // In this case, update the value of min to be equal to max to ensure correct behavior.
    if (maxVal < minVal) {
        min = constructDiopiScalarT(highDtype, maxVal);
    }

    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, highDtype).addConstInput(min, highDtype).addConstInput(max, highDtype).addOutput(out).run();
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
