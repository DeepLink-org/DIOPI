/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../diopi_helper.hpp"

namespace impl {
namespace camb {

extern "C" {
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

static diopiError_t transposeInternal(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor input, int64_t dim0, int64_t dim1) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> cnnlTransposeDesc;
    cnnlTransposeDescriptor_t transposeDesc = cnnlTransposeDesc.get();
    std::vector<int> perms = getPerm(input, dim0, dim1);
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(transposeDesc, perms.size(), perms.data()));

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transposeDesc, &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transposeDesc, inputDesc.get(), input.data(), outDesc.get(), outTensor.data(), workspace, workspaceSize));
    return diopiSuccess;
}

diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t outputSize, diopiSize_t kernelSize,
                         diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    if (inputTensor.shape().size() == 2) {
        inputTensor.unsqueeze(0);
        outTensor.unsqueeze(0);
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_NCHW);

    DiopiTensor inputCol = requiresTensor(ctx, {inputTensor.shape()[0], inputTensor.shape()[2], inputTensor.shape()[1]}, inputTensor.dtype());
    DIOPI_CALL(transposeInternal(ctx, inputCol, inputTensor, 1, 2));
    CnnlTensorDesc inputColDesc(inputCol, CNNL_LAYOUT_ARRAY);

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, inputTensor.dtype()));

    int32_t padHeight = padding.data[0];
    int32_t padWidth = padding.len == 2 ? padding.data[1] : padding.data[0];
    std::vector<int32_t> vPadding = {padHeight, padHeight, padWidth, padWidth};
    int32_t strideHeight = stride.data[0];
    int32_t strideWidth = stride.len == 2 ? stride.data[1] : stride.data[0];
    std::vector<int32_t> vStride = {strideHeight, strideWidth};
    int32_t dilationHeight = dilation.data[0];
    int32_t dilationWidth = dilation.len == 2 ? dilation.data[1] : dilation.data[0];
    std::vector<int32_t> vDilation = {dilationHeight, dilationWidth};
    int32_t kernelSizeHeight = kernelSize.data[0];
    int32_t kernelSizeWidth = kernelSize.len == 2 ? kernelSize.data[1] : kernelSize.data[0];
    int32_t outputSizeHeight = outputSize.data[0];
    int32_t outputSizeWidth = outputSize.len == 2 ? outputSize.data[1] : outputSize.data[0];

    CnnlTensorDesc weightDesc;
    cnnlTensorDescriptor_t wDesc = weightDesc.get();
    std::vector<int> weightSizes = {1, 1, kernelSizeHeight, kernelSizeWidth};
    std::vector<int> weightStrides = {1, 1, 1, 1};
    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(wDesc, CNNL_LAYOUT_NCHW, dtype, weightSizes.size(), weightSizes.data(), weightStrides.data()));

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> cnnlConvDesc;
    cnnlConvolutionDescriptor_t convDesc = cnnlConvDesc.get();
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc, 4, vPadding.data(), vStride.data(), vDilation.data(), 1, dtype));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetCol2ImWorkspaceSize(handle, inputColDesc.get(), wDesc, outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlCol2Im(handle, inputColDesc.get(), inputCol.data(), wDesc, convDesc, workspace, workspaceSize, outDesc.get(), outTensor.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
