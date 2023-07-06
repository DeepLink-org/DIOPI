/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
namespace {

int getDim(int64_t dim, int64_t inputDim) {
    int dimTmp = static_cast<int>(dim);
    if (dimTmp < 0) {
        dimTmp = dimTmp + inputDim;
    }
    return dimTmp;
}

std::vector<uint32_t> getStride(int64_t dim, int64_t inputDim, int64_t step, const int64_t* inputSizes) {
    auto dimTmp = getDim(dim, inputDim);
    std::vector<int64_t> stride(inputDim);
    // fake stride for contiguous input
    int64_t z = 1;
    stride.insert(stride.begin(), z);
    for (int64_t d = inputDim - 1; d > 0; d--) {
        z *= inputSizes[d];
        stride.insert(stride.begin(), z);
    }

    std::vector<int64_t> newStride(inputDim + 1);
    newStride[inputDim] = inputDim == 0 ? 1 : stride[dimTmp];
    for (int64_t d = 0; d < inputDim; d++) {
        auto inputStride = stride[d];
        newStride[d] = (d == dimTmp) ? (step * inputStride) : inputStride;
    }

    std::vector<uint32_t> strides;
    std::transform(newStride.begin(), newStride.end(), std::back_inserter(strides), [](int64_t s) { return static_cast<uint32_t>(s); });
    return strides;
}

}  // namespace

extern "C" {

DIOPI_API diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto inputTensor = DiopiTensor(input);
    auto outTensor = DiopiTensor(out);

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CHECKCNNL(cnnlUnfold(
        handle, static_cast<int>(dim), static_cast<int>(size), static_cast<int>(step), inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t inputSizes,
                                           int64_t dim, int64_t size, int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor gradInputTensor(gradInput);

    std::vector<DiopiTensor*> pTensors{&gradOutputTensor};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float32}));
    DiopiTensor gradOutputTensorTmp = *pTensors[0];
    DiopiTensor gradInputTensorTmp = gradInputTensor;
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensorTmp, gradOutputTensorTmp.dtype()));

    CnnlTensorDesc gradOutputDesc(gradOutputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputTensorTmp, CNNL_LAYOUT_ARRAY);

    uint32_t workspaceSize = 0;
    DIOPI_CHECKCNNL(cnnlGetAsStridedBackwardWorkspaceSize(handle, gradInputDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    auto strides = getStride(dim, inputSizes.len, step, inputSizes.data);

    DIOPI_CHECKCNNL(cnnlAsStridedBackward(
        handle, gradOutputDesc.get(), gradOutputTensorTmp.data(), gradInputDesc.get(), gradInputTensorTmp.data(), strides.data(), 0, workspace, workspaceSize));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
