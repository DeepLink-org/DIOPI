#include <numeric>

#include "../cnnl_helper.hpp"
#include "../diopi_helper.hpp"
#include "common.hpp"

namespace impl {
namespace camb {

static std::vector<int> getPerm(DiopiTensor tensor, int64_t dim0, int64_t dim1) {
    int inputSize = tensor.shape().size();
    if (dim0 < 0) {
        dim0 = dim0 + inputSize;
    }
    if (dim1 < 0) {
        dim1 = dim1 + inputSize;
    }

    std::vector<int> perms(inputSize);
    std::iota(perms.begin(), perms.end(), 0);
    perms[dim0] = dim1;
    perms[dim1] = dim0;
    return perms;
}

// outTensor must have a storage. this funciton doesn't care the stride and the outTenso's shape must be as the same as input's.
diopiError_t transpose(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor input, int64_t dim0, int64_t dim1) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> cnnlTransposeDesc;
    cnnlTransposeDescriptor_t transposeDesc = cnnlTransposeDesc.get();
    std::vector<int> perms = getPerm(input, dim0, dim1);
    cnnlSetTransposeDescriptor(transposeDesc, perms.size(), perms.data());

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> shapeTmp(outTensor.shape().begin(), outTensor.shape().end());
    std::swap(shapeTmp[dim0], shapeTmp[dim1]);
    // calculate the stride.
    int dim = shapeTmp.size();
    int stride = 1;
    std::vector<int32_t> strideTmp(dim);
    for (int i = dim - 1; i >= 0; --i) {
        if (i == dim - 1) {
            strideTmp[i] = 1;
        } else {
            stride = stride * shapeTmp[i + 1];
            strideTmp[i] = stride;
        }
    }
    CnnlTensorDesc outDesc;
    outDesc.set(outTensor.dtype(), shapeTmp, strideTmp, CNNL_LAYOUT_ARRAY);
    size_t workspaceSize = 0;
    cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transposeDesc, &workspaceSize);
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transposeDesc, inputDesc.get(), input.data(), outDesc.get(), outTensor.data(), workspace, workspaceSize));
    return diopiSuccess;
}

diopiError_t transpose(diopiContextHandle_t ctx, const DiopiTensor& inputTensor, DiopiTensor& outTensor, std::vector<int32_t> perms) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> cnnlTransposeDesc;
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(cnnlTransposeDesc.get(), perms.size(), perms.data()));
    if (!outTensor.defined()) {
        std::vector<int64_t> trShape(perms.size());
        for (size_t i = 0; i < perms.size(); ++i) {
            trShape[i] = inputTensor.shape()[perms[i]];
        }
        outTensor = requiresTensor(ctx, trShape, inputTensor.dtype());
    }
    CnnlTensorDesc inDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    size_t workspaceSize = 0;
    cnnlGetTransposeWorkspaceSize(handle, inDesc.get(), cnnlTransposeDesc.get(), &workspaceSize);
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(
        cnnlTranspose_v2(handle, cnnlTransposeDesc.get(), inDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data(), workspace, workspaceSize));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
