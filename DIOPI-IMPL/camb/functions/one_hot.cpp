/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

cnnlCastDataType_t getCastDataType(diopiDtype_t source_dtype, diopiDtype_t dest_dtype) {
    switch (source_dtype) {
        case diopi_dtype_int8:
            switch (dest_dtype) {
                case diopi_dtype_int16:
                    return CNNL_CAST_INT8_TO_INT16;
                case diopi_dtype_int32:
                    return CNNL_CAST_INT8_TO_INT32;
                case diopi_dtype_float16:
                    return CNNL_CAST_INT8_TO_HALF;
                case diopi_dtype_float32:
                    return CNNL_CAST_INT8_TO_FLOAT;
            }
        case diopi_dtype_uint8:
            switch (dest_dtype) {
                case diopi_dtype_int32:
                    return CNNL_CAST_UINT8_TO_INT32;
                case diopi_dtype_int64:
                    return CNNL_CAST_UINT8_TO_INT64;
                case diopi_dtype_float16:
                    return CNNL_CAST_UINT8_TO_HALF;
                case diopi_dtype_float32:
                    return CNNL_CAST_UINT8_TO_FLOAT;
            }
        case diopi_dtype_int16:
            switch (dest_dtype) {
                case diopi_dtype_int32:
                    return CNNL_CAST_INT16_TO_INT32;
                case diopi_dtype_float16:
                    return CNNL_CAST_INT16_TO_HALF;
                case diopi_dtype_float32:
                    return CNNL_CAST_INT16_TO_FLOAT;
            }
        case diopi_dtype_int32:
            switch (dest_dtype) {
                case diopi_dtype_int8:
                    return CNNL_CAST_INT32_TO_INT8;
                case diopi_dtype_int16:
                    return CNNL_CAST_INT32_TO_INT16;
                case diopi_dtype_int64:
                    return CNNL_CAST_INT32_TO_INT64;
                case diopi_dtype_float16:
                    return CNNL_CAST_INT32_TO_HALF;
                case diopi_dtype_float32:
                    return CNNL_CAST_INT32_TO_FLOAT;
                case diopi_dtype_bool:
                    return CNNL_CAST_INT32_TO_BOOL;
            }
        case diopi_dtype_uint32:
            switch (dest_dtype) {
                case diopi_dtype_int64:
                    return CNNL_CAST_UINT32_TO_INT64;
                case diopi_dtype_uint64:
                    return CNNL_CAST_UINT32_TO_UINT64;
            }
        case diopi_dtype_int64:
            switch (dest_dtype) {
                case diopi_dtype_int32:
                    return CNNL_CAST_INT64_TO_INT32;
                case diopi_dtype_uint32:
                    return CNNL_CAST_INT64_TO_UINT32;
                case diopi_dtype_float16:
                    return CNNL_CAST_INT64_TO_HALF;
                case diopi_dtype_float32:
                    return CNNL_CAST_INT64_TO_FLOAT;
            }
        case diopi_dtype_uint64:
            switch (dest_dtype) {
                case diopi_dtype_uint32:
                    return CNNL_CAST_UINT64_TO_UINT32;
            }
        case diopi_dtype_float16:
            switch (dest_dtype) {
                case diopi_dtype_int8:
                    return CNNL_CAST_HALF_TO_INT8;
                case diopi_dtype_uint8:
                    return CNNL_CAST_HALF_TO_UINT8;
                case diopi_dtype_int16:
                    return CNNL_CAST_HALF_TO_INT16;
                case diopi_dtype_int32:
                    return CNNL_CAST_HALF_TO_INT32;
                case diopi_dtype_int64:
                    return CNNL_CAST_HALF_TO_INT64;
                case diopi_dtype_float32:
                    return CNNL_CAST_HALF_TO_FLOAT;
                    // ? CNNL_CAST_HALF_TO_FLOAT_INF
                case diopi_dtype_bool:
                    return CNNL_CAST_HALF_TO_BOOL;
            }
        case diopi_dtype_float32:
            switch (dest_dtype) {
                case diopi_dtype_int8:
                    return CNNL_CAST_FLOAT_TO_INT8;
                case diopi_dtype_uint8:
                    return CNNL_CAST_FLOAT_TO_UINT8;
                case diopi_dtype_int16:
                    return CNNL_CAST_FLOAT_TO_INT16;
                case diopi_dtype_int32:
                    return CNNL_CAST_FLOAT_TO_INT32;
                case diopi_dtype_int64:
                    return CNNL_CAST_FLOAT_TO_INT64;
                case diopi_dtype_float16:
                    return CNNL_CAST_FLOAT_TO_HALF;
                    // ? CNNL_CAST_FLOAT_TO_HALF_IEEE754
                case diopi_dtype_float64:
                    return CNNL_CAST_FLOAT_TO_DOUBLE;
                case diopi_dtype_bool:
                    return CNNL_CAST_FLOAT_TO_BOOL;
            }
        case diopi_dtype_float64:
            switch (dest_dtype) {
                case diopi_dtype_float32:
                    return CNNL_CAST_DOUBLE_TO_FLOAT;
            }
        case diopi_dtype_bool:
            switch (dest_dtype) {
                case diopi_dtype_int32:
                    return CNNL_CAST_BOOL_TO_INT32;
                case diopi_dtype_float16:
                    return CNNL_CAST_BOOL_TO_HALF;
                case diopi_dtype_float32:
                    return CNNL_CAST_BOOL_TO_FLOAT;
            }
    }
}

diopiError_t castDataTypeOut(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t out, diopiDtype_t dtype) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    diopiDtype_t input_dtype;
    diopiGetTensorDtype(input, &input_dtype);
    cnnlCastDataType_t cast_dtype = getCastDataType(input_dtype, dtype);

    DIOPI_CALLCNNL(cnnlCastDataType(handle, input_desc.get(), input_tensor.data(), cast_dtype, out_desc.get(), out_tensor.data()));
    return diopiSuccess;
}

diopiTensorHandle_t castDataType(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiDtype_t dtype) {
    diopiTensorHandle_t out = nullptr;
    diopiSize_t size;
    diopiGetTensorShape(input, &size);
    diopiRequireTensor(ctx, &out, &size, nullptr, dtype, diopi_device);

    castDataTypeOut(ctx, input, out, dtype);
    return out;
}

int32_t maxInt32(diopiContextHandle_t ctx, diopiConstTensorHandle_t input) {
    diopiTensorHandle_t max;
    std::vector<int64_t> dims(1, 1);
    diopiSize_t shape(dims.data(), 1);
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiRequireTensor(ctx, &max, &shape, nullptr, dtype, diopi_device);
    diopiMaxAll(ctx, max, input);

    DiopiTensor max_tensor(max);
    int32_t res = 0;
    int32_t* ptr = reinterpret_cast<int32_t*>(malloc(max_tensor.numel() * sizeof(int32_t)));
    cnrtMemcpy(ptr, max_tensor.data(), max_tensor.numel() * sizeof(int32_t), cnrtMemcpyDevToHost);
    res = *ptr;
    free(ptr);

    return res;
}

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    cnnlDataType_t input_dtype, out_dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&input_dtype, input_tensor.dtype()));
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&out_dtype, out_tensor.dtype()));

    // input must be int32
    diopiTensorHandle_t input32;
    if (CNNL_DTYPE_INT32 != input_dtype) {
        input32 = castDataType(ctx, input, diopi_dtype_int32);
    } else {
        input32 = (diopiTensorHandle_t)input;
    }
    DiopiTensor input32_tensor(input32);
    CnnlTensorDesc input32_desc(input32_tensor, CNNL_LAYOUT_ARRAY);

    if (-1 == numClasses) {
        numClasses = maxInt32(ctx, input32) + 1;
    }

    diopiTensorHandle_t on_value, off_value;
    std::vector<int64_t> dims(1, 1);
    diopiSize_t shape(dims.data(), 1);
    diopiRequireTensor(ctx, &on_value, &shape, nullptr, diopi_dtype_int32, diopi_device);
    diopiRequireTensor(ctx, &off_value, &shape, nullptr, diopi_dtype_int32, diopi_device);
    DiopiTensor on_value_tensor(on_value);
    DiopiTensor off_value_tensor(off_value);
    CnnlTensorDesc on_tensor_desc(on_value_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc off_tensor_desc(off_value_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlFill(handle, 1, on_tensor_desc.get(), on_value_tensor.data()));
    DIOPI_CALLCNNL(cnnlFill(handle, 0, off_tensor_desc.get(), off_value_tensor.data()));
    int axis = -1;

    // output must be int32, float16, float32
    if (CNNL_DTYPE_INT32 != out_dtype && CNNL_DTYPE_HALF != out_dtype && CNNL_DTYPE_FLOAT != out_dtype) {
        diopiTensorHandle_t out32 = castDataType(ctx, out, diopi_dtype_int32);
        DiopiTensor out32_tensor(out32);
        DIOPI_CALLCNNL(cnnlOneHot(handle,
                                  input32_desc.get(),
                                  input32_tensor.data(),
                                  numClasses,
                                  on_value_tensor.data(),
                                  off_value_tensor.data(),
                                  axis,
                                  CNNL_DTYPE_INT32,
                                  out32_tensor.data()));
        castDataTypeOut(ctx, out32, out, out_tensor.dtype());
        return diopiSuccess;
    }

    DIOPI_CALLCNNL(cnnlOneHot(
        handle, input32_desc.get(), input32_tensor.data(), numClasses, on_value_tensor.data(), off_value_tensor.data(), axis, out_dtype, out_tensor.data()));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
