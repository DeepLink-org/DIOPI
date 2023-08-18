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
DIOPI_API diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                   diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    } else if (inputTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_int32));
    }
    DiopiTensor indexTensor(index);
    DIOPI_CALL(autoCastTensorType(ctx, {&indexTensor}, {diopi_dtype_int32, diopi_dtype_int64}));
    DiopiTensor outTensor(out);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim += inputTensor.dim();
    }

    if (outTensor.dtype() == inputTensor.dtype()) {
        DIOPI_CALLCNNL(cnnlGather(handle, dim, inputDesc.get(), inputTensor.data(), indexDesc.get(), indexTensor.data(), outDesc.get(), outTensor.data()));
    } else {
        DiopiTensor outTemp = outTensor;
        DIOPI_CALL(dataTypeCast(ctx, outTemp, inputTensor.dtype()));
        CnnlTensorDesc outTempDesc(outTemp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlGather(handle, dim, inputDesc.get(), inputTensor.data(), indexDesc.get(), indexTensor.data(), outTempDesc.get(), outTemp.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTemp));
    }

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                           diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    diopiScalar_t zero = constructDiopiScalarT(diopi_dtype_float32, 0);
    DIOPI_CALL(diopiFill(ctx, gradInput, &zero));

    DiopiTensor inputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }
    DiopiTensor indexTensor(index);
    DIOPI_CALL(autoCastTensorType(ctx, {&indexTensor}, {diopi_dtype_int32, diopi_dtype_int64}));
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor outTemp = gradInputTensor;
    if (outTemp.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, outTemp, diopi_dtype_float32));
    }
    DiopiTensor gradOutputTensor(gradOutput);
    if (gradOutputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, gradOutputTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTempDesc(outTemp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlScatter(handle,
                               dim,
                               outTempDesc.get(),
                               outTemp.data(),
                               indexDesc.get(),
                               indexTensor.data(),
                               gradOutputDesc.get(),
                               gradOutputTensor.data(),
                               outTempDesc.get(),
                               outTemp.data(),
                               CNNL_SCATTER_ADD));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, outTemp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
