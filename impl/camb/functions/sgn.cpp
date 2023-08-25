/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiSgn(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    if (inputTensor.dtype() == diopi_dtype_complex128 || inputTensor.dtype() == diopi_dtype_complex64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_complex64));
    } else {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }

    DiopiTensor outTensorTemp = outTensor;
    if (inputTensor.dtype() != outTensorTemp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTemp, inputTensor.dtype()));
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlSign(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensorTemp.data()));

    if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSgnInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(diopiSgn(ctx, input, input));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
