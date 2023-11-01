/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiLerpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                             diopiConstTensorHandle_t weight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    DiopiTensor endTensor(end);
    DiopiTensor weightTensor(weight);

    DiopiTensor outTensorTmp = outTensor;
    DiopiTensor inputTensorTmp = inputTensor;
    DiopiTensor endTensorTmp = endTensor;
    DiopiTensor weightTensorTmp = weightTensor;

    std::vector<DiopiTensor *> tensors{&inputTensorTmp, &endTensorTmp, &weightTensorTmp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, supportedDtypes));
    if (inputTensorTmp.dtype() != outTensor.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), inputTensorTmp.dtype());
    }

    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc endDesc(endTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc weightDesc(weightTensorTmp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetLerpWorkspaceSize(handle, inputDesc.get(), endDesc.get(), weightDesc.get(), outDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlLerp(handle,
                            inputDesc.get(),
                            inputTensorTmp.data(),
                            endDesc.get(),
                            endTensorTmp.data(),
                            weightDesc.get(),
                            weightTensorTmp.data(),
                            workspace,
                            workspaceSize,
                            outDesc.get(),
                            outTensorTmp.data()));
    if (outTensor.dtype() != outTensorTmp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}
diopiError_t diopiLerpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                             const diopiScalar_t *weight) {
    DiopiTensor weightTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, weight, weightTensor));
    DIOPI_CALL(diopiLerpTensor(ctx, out, input, end, static_cast<diopiTensorHandle_t>(weightTensor)));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
