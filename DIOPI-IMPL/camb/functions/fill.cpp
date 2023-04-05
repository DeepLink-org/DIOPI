/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/float16.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    DiopiTensor input_tensor_temp = input_tensor;

    // float64 not supported yet
    if (input_tensor.dtype() == diopi_dtype_float64) {
       dataTypeCast(ctx, input_tensor_temp, diopi_dtype_float32);
    }

    CnnlTensorDesc input_tensor_desc(input_tensor_temp, CNNL_LAYOUT_ARRAY);

    double value_scalar = DiopiDataType::isInteger(value->stype) ? value->ival : value->fval;
    void* value_ptr;
    switch (input_tensor_temp.dtype()) {
        case diopi_dtype_bool: {
            auto temp = static_cast<bool>(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_int8: {
            auto temp = int8_t(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_uint8: {
            auto temp = uint8_t(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_int16: {
            auto temp = int16_t(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_uint16: {
            auto temp = int16_t(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_int32: {
            auto temp = int32_t(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_uint32: {
            auto temp = uint32_t(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_int64: {
            auto temp = int64_t(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_uint64: {
            auto temp = uint64_t(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_float16: {
            auto temp = half_float::half(value_scalar);
            value_ptr = &temp;
            break;
        }
        case diopi_dtype_float32: {
            auto temp = static_cast<float>(value_scalar);
            value_ptr = &temp;
            break;
        }
    }

    DIOPI_CALLCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, value_ptr, input_tensor_desc.get(), input_tensor_temp.data()));

    if (input_tensor_temp.dtype() != input_tensor.dtype()) {
        dataTypeCast(ctx, input_tensor, input_tensor_temp);
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
