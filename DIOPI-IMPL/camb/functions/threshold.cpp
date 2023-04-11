#include <diopi/functions.h>

#include "../../third_party/half/include/half.hpp"
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

DIOPI_API diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                                      const diopiScalar_t* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);

    std::vector<DiopiTensor*> pTensors{&input_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int8, diopi_dtype_uint8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor out_tensor_temp = out_tensor;
    if (out_tensor.dtype() != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_temp, input_tensor.dtype()));
    }

    CnnlTensorDesc input_tensor_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_tensor_desc(out_tensor_temp, CNNL_LAYOUT_ARRAY);

    auto threshold_scalar = DiopiDataType::isInteger(threshold->stype) ? threshold->ival : threshold->fval;
    auto value_scalar = DiopiDataType::isInteger(value->stype) ? value->ival : value->fval;

    void* threshold_val;
    void* value_val;
    int8_t value_int8, threshold_val_int8;
    uint8_t value_uint8, threshold_val_uint8;
    int16_t value_int16, threshold_val_int16;
    int32_t value_int32, threshold_val_int32;
    half_float::half value_float16, threshold_val_float16;
    float value_float32, threshold_val_float32;

    switch (input_tensor.dtype()) {
        case diopi_dtype_int8: {
            threshold_val_int8 = int8_t(threshold_scalar);
            value_int8 = int8_t(value_scalar);
            threshold_val = &threshold_val_int8;
            value_val = &value_int8;
            break;
        }
        case diopi_dtype_uint8: {
            threshold_val_uint8 = uint8_t(threshold_scalar);
            value_uint8 = uint(value_scalar);
            threshold_val = &threshold_val_uint8;
            value_val = &value_uint8;
            break;
        }
        case diopi_dtype_int16: {
            threshold_val_int16 = int16_t(threshold_scalar);
            value_int16 = int16_t(value_scalar);
            threshold_val = &threshold_val_int16;
            value_val = &value_int16;
            break;
        }
        case diopi_dtype_int32: {
            threshold_val_int32 = int32_t(threshold_scalar);
            value_int32 = int32_t(value_scalar);
            threshold_val = &threshold_val_int32;
            value_val = &value_int32;
            break;
        }
        case diopi_dtype_float16: {
            threshold_val_float16 = half_float::half(threshold_scalar);
            value_float16 = half_float::half(value_scalar);
            threshold_val = &threshold_val_float16;
            value_val = &value_float16;
            break;
        }
        case diopi_dtype_float32: {
            threshold_val_float32 = static_cast<float>(threshold_scalar);
            value_float32 = static_cast<float>(value_scalar);
            threshold_val = &threshold_val_float32;
            value_val = &value_float32;
            break;
        }
        default:
            break;
    }

    DIOPI_CALLCNNL(
        cnnlThreshold(handle, input_tensor_desc.get(), input_tensor.data(), threshold_val, value_val, out_tensor_desc.get(), out_tensor_temp.data()));

    if (out_tensor_temp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_temp));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    diopiThreshold(ctx, input, input, threshold, value);
}

DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);

    std::vector<DiopiTensor*> pTensors{&input_tensor, &grad_output_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor grad_input_tensor_temp = grad_input_tensor;
    if (grad_input_tensor.dtype() != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor_temp, input_tensor.dtype()));
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_input_desc(grad_input_tensor_temp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_output_desc(grad_output_tensor, CNNL_LAYOUT_ARRAY);

    double threshold_scalar = DiopiDataType::isInteger(threshold->stype) ? threshold->ival : threshold->fval;

    void* threshold_val;
    half_float::half threshold_scalar_half;
    float threshold_scalar_float;
    switch (input_tensor.dtype()) {
        case diopi_dtype_float16: {
            threshold_scalar_half = half_float::half(threshold_scalar);
            threshold_val = &threshold_scalar_half;
            break;
        }
        case diopi_dtype_float32: {
            threshold_scalar_float = static_cast<float>(threshold_scalar);
            threshold_val = &threshold_scalar_float;
            break;
        }
        default:
            break;
    }

    DIOPI_CALLCNNL(cnnlThresholdBackward(handle,
                                         input_desc.get(),
                                         input_tensor.data(),
                                         grad_output_desc.get(),
                                         grad_output_tensor.data(),
                                         threshold_val,
                                         grad_input_desc.get(),
                                         grad_input_tensor_temp.data()))

    if (grad_input_tensor_temp.dtype() != grad_input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, grad_input_tensor_temp));
    }
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
