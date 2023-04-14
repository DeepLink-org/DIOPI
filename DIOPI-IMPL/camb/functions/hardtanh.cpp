/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val,
                                     const diopiScalar_t* max_val) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor out_tensor(out);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    float min = min_val->fval;
    float max = max_val->fval;
    if (min > max) {
        min = max;
    }

    if (out_tensor.dtype() == diopi_dtype_float64) {
        DiopiTensor out32_tensor = requiresTensor(ctx, out_tensor.shape(), diopi_dtype_float32);
        CnnlTensorDesc out32Desc(out32_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), input_tensor.data(), max, min, out32Desc.get(), out32_tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out32_tensor));
    } else {
        DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), input_tensor.data(), max, min, outDesc.get(), out_tensor.data()));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor out_tensor(input);

    float min = min_val->fval;
    float max = max_val->fval;
    if (min > max) {
        min = max;
    }

    if (out_tensor.dtype() == diopi_dtype_float64) {
        DiopiTensor out32_tensor = requiresTensor(ctx, input_tensor.shape(), diopi_dtype_float32);
        CnnlTensorDesc out32Desc(out32_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), input_tensor.data(), max, min, out32Desc.get(), out32_tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out32_tensor));
    } else {
        DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), input_tensor.data(), max, min, inputDesc.get(), input_tensor.data()));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor grad_out_tensor(grad_output);
    if (grad_out_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_out_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc gradoutDesc(grad_out_tensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor grad_in_tensor(grad_input);
    CnnlTensorDesc gradinDesc(grad_in_tensor, CNNL_LAYOUT_ARRAY);

    float min = min_val->fval;
    float max = max_val->fval;
    if (min > max) {
        min = max;
    }

    if (grad_in_tensor.dtype() == diopi_dtype_float64) {
        DiopiTensor grad_in32_tensor = requiresTensor(ctx, grad_in_tensor.shape(), diopi_dtype_float32);
        CnnlTensorDesc gradin32Desc(grad_in32_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlHardtanhBackward(
            handle, inputDesc.get(), input_tensor.data(), gradoutDesc.get(), grad_out_tensor.data(), max, min, gradin32Desc.get(), grad_in32_tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, grad_in_tensor, grad_in32_tensor));
    } else {
        DIOPI_CALLCNNL(cnnlHardtanhBackward(
            handle, inputDesc.get(), input_tensor.data(), gradoutDesc.get(), grad_out_tensor.data(), max, min, gradinDesc.get(), grad_in_tensor.data()));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
