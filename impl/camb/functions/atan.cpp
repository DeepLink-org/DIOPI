/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

static diopiError_t atan(diopiContextHandle_t ctx, DiopiTensor output, DiopiTensor input1) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<DiopiTensor *> tensor{&input1};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensor, supportedDtypes));
    DiopiTensor input2 = ones(ctx, input1.shape(), input1.dtype());

    DiopiTensor outputTmp = output;
    if (input1.dtype() != output.dtype()) {
        outputTmp = requiresTensor(ctx, output.shape(), input1.dtype());
    }

    CnnlTensorDesc input1Desc(input1, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc input2Desc(input2, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTmp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAtan2WorkspaceSize(handle, input1Desc.get(), input2Desc.get(), outputDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlAtan2(handle,
                             CNNL_COMPUTATION_HIGH_PRECISION,
                             input1Desc.get(),
                             input1.data(),
                             input2Desc.get(),
                             input2.data(),
                             workspace,
                             workspaceSize,
                             outputDesc.get(),
                             outputTmp.data()));
    if (outputTmp.dtype() != output.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output, outputTmp));
    }
    return diopiSuccess;
}

diopiError_t diopiAtan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    DIOPI_CALL(atan(ctx, outTensor, inputTensor));
    return diopiSuccess;
}

diopiError_t diopiAtanInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor outTensor(input);
    DiopiTensor inputTensor(input);
    DIOPI_CALL(atan(ctx, outTensor, inputTensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
