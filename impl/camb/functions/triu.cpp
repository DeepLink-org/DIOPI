/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);

    DiopiTensor outTensorTmp = outTensor;
    DiopiTensor inputTensorTmp = inputTensor;

    std::vector<DiopiTensor *> tensor{&inputTensorTmp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensor, supportedDtypes));
    if (inputTensorTmp.dtype() != outTensor.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), inputTensorTmp.dtype());
    }

    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlTri(handle, static_cast<int32_t>(diagonal), true, inputDesc.get(), inputTensorTmp.data(), outDesc.get(), outTensorTmp.data()));
    if (outTensor.dtype() != outTensorTmp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}
diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
    DIOPI_CALL(diopiTriu(ctx, input, input, diagonal));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
