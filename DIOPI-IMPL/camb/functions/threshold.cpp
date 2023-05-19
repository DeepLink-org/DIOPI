#include <diopi/functions.h>

#include "../../third_party/half/include/half.hpp"
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                                      const diopiScalar_t* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> pTensors{&inputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int8, diopi_dtype_uint8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outTensorTemp = outTensor;
    if (outTensor.dtype() != inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTemp, inputTensor.dtype()));
    }

    CnnlTensorDesc inputTensorDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTensorDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);

    auto thresholdScalar = DiopiDataType::isInteger(threshold->stype) ? threshold->ival : threshold->fval;
    auto valueScalar = DiopiDataType::isInteger(value->stype) ? value->ival : value->fval;

    void* thresholdVal;
    void* valueVal;
    int8_t valueInt8, thresholdValInt8;
    uint8_t valueUint8, thresholdValUint8;
    int16_t valueInt16, thresholdValInt16;
    int32_t valueInt32, thresholdValInt32;
    half_float::half valueFloat16, thresholdValFloat16;
    float valueFloat32, thresholdValFloat32;

    switch (inputTensor.dtype()) {
        case diopi_dtype_int8: {
            thresholdValInt8 = int8_t(thresholdScalar);
            valueInt8 = int8_t(valueScalar);
            thresholdVal = &thresholdValInt8;
            valueVal = &valueInt8;
            break;
        }
        case diopi_dtype_uint8: {
            thresholdValUint8 = uint8_t(thresholdScalar);
            valueUint8 = uint(valueScalar);
            thresholdVal = &thresholdValUint8;
            valueVal = &valueUint8;
            break;
        }
        case diopi_dtype_int16: {
            thresholdValInt16 = int16_t(thresholdScalar);
            valueInt16 = int16_t(valueScalar);
            thresholdVal = &thresholdValInt16;
            valueVal = &valueInt16;
            break;
        }
        case diopi_dtype_int32: {
            thresholdValInt32 = int32_t(thresholdScalar);
            valueInt32 = int32_t(valueScalar);
            thresholdVal = &thresholdValInt32;
            valueVal = &valueInt32;
            break;
        }
        case diopi_dtype_float16: {
            thresholdValFloat16 = half_float::half(thresholdScalar);
            valueFloat16 = half_float::half(valueScalar);
            thresholdVal = &thresholdValFloat16;
            valueVal = &valueFloat16;
            break;
        }
        case diopi_dtype_float32: {
            thresholdValFloat32 = static_cast<float>(thresholdScalar);
            valueFloat32 = static_cast<float>(valueScalar);
            thresholdVal = &thresholdValFloat32;
            valueVal = &valueFloat32;
            break;
        }
        default:
            break;
    }

    DIOPI_CALLCNNL(
        cnnlThreshold(handle, inputTensorDesc.get(), inputTensor.data(), thresholdVal, valueVal, outTensorDesc.get(), outTensorTemp.data()));

    if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}

diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    DIOPI_CALL(diopiThreshold(ctx, input, input, threshold, value));
    return diopiSuccess;
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &gradOutputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor gradInputTensorTemp = gradInputTensor;
    if (gradInputTensor.dtype() != inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensorTemp, inputTensor.dtype()));
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputTensorTemp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_ARRAY);

    double thresholdScalar = DiopiDataType::isInteger(threshold->stype) ? threshold->ival : threshold->fval;

    void* thresholdVal;
    half_float::half thresholdScalarHalf;
    float thresholdScalarFloat;
    switch (inputTensor.dtype()) {
        case diopi_dtype_float16: {
            thresholdScalarHalf = half_float::half(thresholdScalar);
            thresholdVal = &thresholdScalarHalf;
            break;
        }
        case diopi_dtype_float32: {
            thresholdScalarFloat = static_cast<float>(thresholdScalar);
            thresholdVal = &thresholdScalarFloat;
            break;
        }
        default:
            break;
    }

    DIOPI_CALLCNNL(cnnlThresholdBackward(handle,
                                         inputDesc.get(),
                                         inputTensor.data(),
                                         gradOutputDesc.get(),
                                         gradOutputTensor.data(),
                                         thresholdVal,
                                         gradInputDesc.get(),
                                         gradInputTensorTemp.data()))

    if (gradInputTensorTemp.dtype() != gradInputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTemp));
    }
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
