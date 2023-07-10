/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/float16.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor inputTensorTemp = inputTensor;

    // float64 not supported yet
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensorTemp, diopi_dtype_float32));
    }

    CnnlTensorDesc inputTensorDesc(inputTensorTemp, CNNL_LAYOUT_ARRAY);

    double valueScalar = DiopiDataType::isInteger(value->stype) ? value->ival : value->fval;
    void* valuePtr = nullptr;
    bool tempBool = false;
    int8_t tempI8 = 0;
    uint8_t tempU8 = 0;
    int16_t tempI16 = 0;
    uint16_t tempU16 = 0;
    int32_t tempI32 = 0;
    uint32_t tempU32 = 0;
    int64_t tempI64 = 0;
    uint64_t tempU64 = 0;
    half_float::half tempF16 = static_cast<half_float::half>(0);
    float tempF32 = 0;

    switch (inputTensorTemp.dtype()) {
        case diopi_dtype_bool: {
            tempBool = static_cast<bool>(valueScalar);
            valuePtr = &tempBool;
            break;
        }
        case diopi_dtype_int8: {
            tempI8 = int8_t(valueScalar);
            valuePtr = &tempI8;
            break;
        }
        case diopi_dtype_uint8: {
            tempU8 = uint8_t(valueScalar);
            valuePtr = &tempU8;
            break;
        }
        case diopi_dtype_int16: {
            tempI16 = int16_t(valueScalar);
            valuePtr = &tempI16;
            break;
        }
        case diopi_dtype_uint16: {
            tempU16 = uint16_t(valueScalar);
            valuePtr = &tempU16;
            break;
        }
        case diopi_dtype_int32: {
            tempI32 = int32_t(valueScalar);
            valuePtr = &tempI32;
            break;
        }
        case diopi_dtype_uint32: {
            tempU32 = uint32_t(valueScalar);
            valuePtr = &tempU32;
            break;
        }
        case diopi_dtype_int64: {
            tempI64 = int64_t(valueScalar);
            valuePtr = &tempI64;
            break;
        }
        case diopi_dtype_uint64: {
            tempU64 = uint64_t(valueScalar);
            valuePtr = &tempU64;
            break;
        }
        case diopi_dtype_float16: {
            tempF16 = half_float::half(valueScalar);
            valuePtr = &tempF16;
            break;
        }
        case diopi_dtype_float32: {
            tempF32 = static_cast<float>(valueScalar);
            valuePtr = &tempF32;
            break;
        }
        default: {
            DIOPI_CHECK(false, "the input tensor dtype %s is not allown", DiopiDataType::dataTypeStr(inputTensorTemp.dtype()).c_str());
        }
    }

    DIOPI_CALLCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, valuePtr, inputTensorDesc.get(), inputTensorTemp.data()));

    if (inputTensorTemp.dtype() != inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, inputTensorTemp));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
