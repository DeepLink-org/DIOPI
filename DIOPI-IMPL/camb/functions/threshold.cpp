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
    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(out);

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
    switch (input_tensor.dtype()) {
        case diopi_dtype_int8: {
            auto temp1 = int8_t(threshold_scalar);
            auto temp2 = int8_t(value_scalar);
            threshold_val = &temp1;
            value_val = &temp2;
            break;
        }
        case diopi_dtype_uint8: {
            auto temp1 = uint8_t(threshold_scalar);
            auto temp2 = uint(value_scalar);
            threshold_val = &temp1;
            value_val = &temp2;
            break;
        }
        case diopi_dtype_int16: {
            auto temp1 = int16_t(threshold_scalar);
            auto temp2 = int16_t(value_scalar);
            threshold_val = &temp1;
            value_val = &temp2;
            break;
        }
        case diopi_dtype_int32: {
            auto temp1 = int32_t(threshold_scalar);
            auto temp2 = int32_t(value_scalar);
            threshold_val = &temp1;
            value_val = &temp2;
            break;
        }
        case diopi_dtype_float16: {
            auto temp1 = half_float::half(threshold_scalar);
            auto temp2 = half_float::half(value_scalar);
            threshold_val = &temp1;
            value_val = &temp2;
            break;
        }
        case diopi_dtype_float32: {
            auto temp1 = static_cast<float>(threshold_scalar);
            auto temp2 = static_cast<float>(value_scalar);
            threshold_val = &temp1;
            value_val = &temp2;
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
    auto input_tensor = DiopiTensor(input);
    auto grad_input_tensor = DiopiTensor(grad_input);
    auto grad_output_tensor = DiopiTensor(grad_output);

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
    switch (input_tensor.dtype()) {
        case diopi_dtype_float16: {
            auto temp = half_float::half(threshold_scalar);
            threshold_val = &temp;
            break;
        }
        case diopi_dtype_float32: {
            auto temp = static_cast<float>(threshold_scalar);
            threshold_val = &temp;
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
