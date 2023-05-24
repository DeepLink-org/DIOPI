/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<CnnlTensorDesc> inputsDesc(numInputs);
    std::vector<cnnlTensorDescriptor_t> inputsDescTmp(numInputs);
    std::vector<const void *> inputs(numInputs);
    for (int i = 0; i < numInputs; i++) {
        DiopiTensor tempTensor(tensors[i]);
        inputsDesc[i].set(tempTensor, CNNL_LAYOUT_ARRAY);
        inputsDescTmp[i] = inputsDesc[i].get();
        inputs[i] = tempTensor.data();
    }

    size_t workspaceSize(0);
    DIOPI_CALLCNNL(cnnlGetConcatWorkspaceSize(handle, numInputs, &workspaceSize));
    void * workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DiopiTensor outTensor(out);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlConcat(handle, numInputs, dim, inputsDescTmp.data(), inputs.data(), workspace, workspaceSize, outDesc.get(), outTensor.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
