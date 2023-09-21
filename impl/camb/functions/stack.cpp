

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {
diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    std::vector<CnnlTensorDesc> inputsDesc(numTensors);
    std::vector<cnnlTensorDescriptor_t> inputsDescTmp(numTensors);
    std::vector<const void*> inputsData(numTensors);

    dim += dim < 0 ? DiopiTensor(tensors[0]).shape().size() + 1 : 0;

    // insert a new dim to input_tensors
    for (int i = 0; i < numTensors; i++) {
        DiopiTensor tempTensor(tensors[i]);
        std::vector<int> catShape(tempTensor.shape().begin(), tempTensor.shape().end());
        cnnlDataType_t dtype;
        CnnlDataType::convertToCnnlType(&dtype, tempTensor.dtype());
        catShape.insert(catShape.begin() + dim, 1);
        int catDimNb = catShape.size();
        inputsData[i] = tempTensor.data();
        inputsDesc[i].set(tempTensor, CNNL_LAYOUT_ARRAY);
        inputsDescTmp[i] = inputsDesc[i].get();
        DIOPI_CALL_CNNL(cnnlSetTensorDescriptor(inputsDescTmp[i], CNNL_LAYOUT_ARRAY, dtype, catDimNb, catShape.data()));
    }
    size_t workspaceSize(0);
    DIOPI_CALL_CNNL(cnnlGetConcatWorkspaceSize(handle, numTensors, &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DiopiTensor outTensor(out);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALL_CNNL(cnnlConcat(handle, numTensors, dim, inputsDescTmp.data(), inputsData.data(), workspace, workspaceSize, outDesc.get(), outTensor.data()));
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
