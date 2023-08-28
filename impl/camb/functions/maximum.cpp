/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    DiopiTensor otherTensor(other);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &otherTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor outTensorTmp = outputTensor;
    DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensor.dtype()));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetMaximumWorkspaceSize(handle, outputDesc.get(), &workspaceSize));

    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlMaximum(
        handle, inputDesc.get(), inputTensor.data(), otherDesc.get(), otherTensor.data(), outputDesc.get(), outTensorTmp.data(), workspacePtr, workspaceSize));

    DIOPI_CALL(dataTypeCast(ctx, outputTensor, outTensorTmp));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
