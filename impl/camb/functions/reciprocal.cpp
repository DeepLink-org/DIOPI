/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    diopiDtype_t originDtype = inputTensor.dtype();
    std::vector<DiopiTensor*> pTensors{&inputTensor, &outTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALL_CNNL(cnnlReciprocal(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));

    if (originDtype == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, originDtype));
    }
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiReciprocal(ctx, input, input);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
