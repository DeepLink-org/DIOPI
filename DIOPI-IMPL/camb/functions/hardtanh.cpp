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

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* minVal,
                                     const diopiScalar_t* maxVal) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor outTensor(out);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    float min = minVal->fval;
    float max = maxVal->fval;
    if (min > max) {
        min = max;
    }

    if (outTensor.dtype() == diopi_dtype_float64) {
        DiopiTensor out32Tensor = requiresTensor(ctx, outTensor.shape(), diopi_dtype_float32);
        CnnlTensorDesc out32Desc(out32Tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), inputTensor.data(), max, min, out32Desc.get(), out32Tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, out32Tensor));
    } else {
        DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), inputTensor.data(), max, min, outDesc.get(), outTensor.data()));
    }
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor outTensor(input);

    float min = minVal->fval;
    float max = maxVal->fval;
    if (min > max) {
        min = max;
    }

    if (outTensor.dtype() == diopi_dtype_float64) {
        DiopiTensor out32Tensor = requiresTensor(ctx, inputTensor.shape(), diopi_dtype_float32);
        CnnlTensorDesc out32Desc(out32Tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), inputTensor.data(), max, min, out32Desc.get(), out32Tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, out32Tensor));
    } else {
        DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), inputTensor.data(), max, min, inputDesc.get(), inputTensor.data()));
    }
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor gradOutTensor(gradOutput);
    if (gradOutTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, gradOutTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc gradoutDesc(gradOutTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor gradInTensor(gradInput);
    CnnlTensorDesc gradinDesc(gradInTensor, CNNL_LAYOUT_ARRAY);

    float min = minVal->fval;
    float max = maxVal->fval;
    if (min > max) {
        min = max;
    }

    if (gradInTensor.dtype() == diopi_dtype_float64) {
        DiopiTensor gradIn32Tensor = requiresTensor(ctx, gradInTensor.shape(), diopi_dtype_float32);
        CnnlTensorDesc gradin32Desc(gradIn32Tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlHardtanhBackward(
            handle, inputDesc.get(), inputTensor.data(), gradoutDesc.get(), gradOutTensor.data(), max, min, gradin32Desc.get(), gradIn32Tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, gradInTensor, gradIn32Tensor));
    } else {
        DIOPI_CALLCNNL(cnnlHardtanhBackward(
            handle, inputDesc.get(), inputTensor.data(), gradoutDesc.get(), gradOutTensor.data(), max, min, gradinDesc.get(), gradInTensor.data()));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
