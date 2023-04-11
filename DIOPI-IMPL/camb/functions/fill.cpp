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
    DiopiTensor input_tensor(input);
    DiopiTensor input_tensor_temp = input_tensor;

    // float64 not supported yet
    if (input_tensor.dtype() == diopi_dtype_float64) {
       DIOPI_CALL(dataTypeCast(ctx, input_tensor_temp, diopi_dtype_float32));
    }

    CnnlTensorDesc input_tensor_desc(input_tensor_temp, CNNL_LAYOUT_ARRAY);

    double value_scalar = DiopiDataType::isInteger(value->stype) ? value->ival : value->fval;
    void* value_ptr;
    bool temp_bool = 0;
    int8_t temp_i8 = 0;
    uint8_t temp_u8 = 0;
    int16_t temp_i16 = 0;
    uint16_t temp_u16 = 0;
    int32_t temp_i32 = 0;
    uint32_t temp_u32 = 0;
    int64_t temp_i64 = 0;
    uint64_t temp_u64 = 0;
    half_float::half temp_f16 = static_cast<half_float::half>(0);
    float temp_f32 = 0;

    switch (input_tensor_temp.dtype()) {
        case diopi_dtype_bool: {
            temp_bool = static_cast<bool>(value_scalar);
            value_ptr = &temp_bool;
            break;
        }
        case diopi_dtype_int8: {
            temp_i8 = int8_t(value_scalar);
            value_ptr = &temp_i8;
            break;
        }
        case diopi_dtype_uint8: {
            temp_u8 = uint8_t(value_scalar);
            value_ptr = &temp_u8;
            break;
        }
        case diopi_dtype_int16: {
            temp_i16 = int16_t(value_scalar);
            value_ptr = &temp_i16;
            break;
        }
        case diopi_dtype_uint16: {
            temp_u16 = uint16_t(value_scalar);
            value_ptr = &temp_u16;
            break;
        }
        case diopi_dtype_int32: {
            temp_i32 = int32_t(value_scalar);
            value_ptr = &temp_i32;
            break;
        }
        case diopi_dtype_uint32: {
            temp_u32 = uint32_t(value_scalar);
            value_ptr = &temp_u32;
            break;
        }
        case diopi_dtype_int64: {
            temp_i64 = int64_t(value_scalar);
            value_ptr = &temp_i64;
            break;
        }
        case diopi_dtype_uint64: {
            temp_u64 = uint64_t(value_scalar);
            value_ptr = &temp_u64;
            break;
        }
        case diopi_dtype_float16: {
            temp_f16 = half_float::half(value_scalar);
            value_ptr = &temp_f16;
            break;
        }
        case diopi_dtype_float32: {
            temp_f32 = static_cast<float>(value_scalar);
            value_ptr = &temp_f32;
            break;
        }
    }

    DIOPI_CALLCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, value_ptr, input_tensor_desc.get(), input_tensor_temp.data()));

    if (input_tensor_temp.dtype() != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, input_tensor_temp));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
