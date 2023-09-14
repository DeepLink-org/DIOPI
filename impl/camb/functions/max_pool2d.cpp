/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstring>
#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

std::pair<int64_t, int64_t> extractDims(const diopiSize_t& size) {
    int64_t dimH = size.data[0];
    int64_t dimW = size.len == 1 ? dimH : size.data[1];
    return {dimH, dimW};
}

diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernelSize, diopiSize_t stride,
                            diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTr(input);
    DiopiTensor outTensor(out);

    DIOPI_CHECK(inputTr.dim() == 3 || inputTr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&inputTr};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));

    if (inputTr.dim() == 3) {
        inputTr.unsqueeze(0);
        outTensor.unsqueeze(0);
    }

    DiopiTensor outTmpTensor = outTensor;
    if (inputTr.dtype() != outTensor.dtype()) {
        outTmpTensor = requiresTensor(ctx, outTensor.shape(), inputTr.dtype());
    }

    std::vector<int64_t> inputDim = inputTr.shape();
    std::vector<int64_t> outDim = outTmpTensor.shape();
    CnnlTensorDesc inputDesc(inputTr, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc outDesc(outTmpTensor, CNNL_LAYOUT_NCHW);

    // structured bindings, introduced in c++17
    auto[kernelH, kernelW] = extractDims(kernelSize);
    auto[strideH, strideW] = stride.len == 0 ? std::make_pair(kernelH, kernelW) : extractDims(stride);
    auto[padH, padW] = extractDims(padding);
    auto[dilation0, dilation1] = extractDims(dilation);

    DIOPI_CHECK(dilation0 == 1 && dilation1 == 1, "Camb kernel only support dilation == 1");

    // calculate padding coefficients
    auto padLeft = padW, padRight = padW, padUp = padH, padDown = padH;
    if (ceilMode) {
        // diff = (out - 1) * stride + kernel_size - input
        int diffHeight = (outDim[2] - 1) * strideH + kernelH - inputDim[2];
        int diffWidth = (outDim[3] - 1) * strideW + kernelW - inputDim[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        padDown = diffHeight > padH ? diffHeight - padH : 0;
        padRight = diffWidth > padW ? diffWidth - padW : 0;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> cnnlPoolDesc;
    cnnlPoolingDescriptor_t poolDesc = cnnlPoolDesc.get();
    DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
        poolDesc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernelH, kernelW, padUp, padDown, padLeft, padRight, strideH, strideW, dilation0, dilation1, ceilMode));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetPoolingWorkspaceSize(handle, CNNL_POOLING_MAX, outTensor.shape()[3], inputTr.shape()[2], &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlPoolingForward_v2(
        handle, poolDesc, nullptr, inputDesc.get(), inputTr.data(), nullptr, nullptr, outDesc.get(), outTmpTensor.data(), workspacePtr, workspaceSize));

    if (outTmpTensor.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTmpTensor));
    }

    return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTr(input);
    DiopiTensor outTensor(out);
    DiopiTensor indicesTr(indices);

    DIOPI_CHECK(inputTr.dim() == 3 || inputTr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&inputTr};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));

    std::vector<DiopiTensor*> pTensorsIndices{&indicesTr};
    DIOPI_CALL(autoCastTensorType(ctx, pTensorsIndices, {diopi_dtype_int16, diopi_dtype_int32}));

    if (inputTr.dim() == 3) {
        inputTr.unsqueeze(0);
        indicesTr.unsqueeze(0);
        outTensor.unsqueeze(0);
    }

    DiopiTensor outTmpTensor = outTensor;
    if (inputTr.dtype() != outTensor.dtype()) {
        outTmpTensor = requiresTensor(ctx, outTensor.shape(), inputTr.dtype());
    }

    std::vector<int64_t> inputDim = inputTr.shape();
    std::vector<int64_t> indicesDim = indicesTr.shape();
    std::vector<int64_t> outDim = outTmpTensor.shape();
    CnnlTensorDesc inputDesc(inputTr, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc indicesDesc(indicesTr, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc outDesc(outTmpTensor, CNNL_LAYOUT_NCHW);

    auto[kernelH, kernelW] = extractDims(kernelSize);
    auto[strideH, strideW] = stride.len == 0 ? std::make_pair(kernelH, kernelW) : extractDims(stride);
    auto[padH, padW] = extractDims(padding);
    auto[dilation0, dilation1] = extractDims(dilation);

    DIOPI_CHECK(dilation0 == 1 && dilation1 == 1, "Camb kernel only support dilation == 1");

    // calculate padding coefficients
    auto padLeft = padW, padRight = padW, padUp = padH, padDown = padH;
    if (ceilMode) {
        // diff = (out - 1) * stride + kernel_size - input
        int diffHeight = (outDim[2] - 1) * strideH + kernelH - inputDim[2];
        int diffWidth = (outDim[3] - 1) * strideW + kernelW - inputDim[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        padDown = diffHeight > padH ? diffHeight - padH : 0;
        padRight = diffWidth > padW ? diffWidth - padW : 0;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> cnnlPoolDesc;
    cnnlPoolingDescriptor_t poolDesc = cnnlPoolDesc.get();
    int poolRank = kernelSize.len;
    if (poolRank == 3) {
        std::vector<int> window{kernelSize.data, kernelSize.data + kernelSize.len};
        std::vector<int> paddingTmp{padding.data, padding.data + padding.len};
        std::vector<int> strideTmp{stride.data, stride.data + stride.len};
        std::vector<int> dilationTmp{dilation.data, dilation.data + dilation.len};
        DIOPI_CALLCNNL(cnnlSetPoolingNdDescriptor_v2(
            poolDesc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, poolRank + 2, window.data(), paddingTmp.data(), strideTmp.data(), dilationTmp.data(), ceilMode));
    } else {
        DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(poolDesc,
                                                     CNNL_POOLING_MAX,
                                                     CNNL_PROPAGATE_NAN,
                                                     kernelH,
                                                     kernelW,
                                                     padUp,
                                                     padDown,
                                                     padLeft,
                                                     padRight,
                                                     strideH,
                                                     strideW,
                                                     dilation0,
                                                     dilation1,
                                                     ceilMode));
    }

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetPoolingWithIndexWorkspaceSize(handle, inputDesc.get(), outDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlPoolingForwardWithIndex(handle,
                                               poolDesc,
                                               nullptr,
                                               inputDesc.get(),
                                               inputTr.data(),
                                               nullptr,
                                               outDesc.get(),
                                               outTmpTensor.data(),
                                               indicesDesc.get(),
                                               indicesTr.data(),
                                               workspacePtr,
                                               workspaceSize));

    if (outTmpTensor.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTmpTensor));
    }

    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                    bool ceilMode, diopiConstTensorHandle_t indices) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTr(input);
    DiopiTensor gradInputTr(gradInput);
    DiopiTensor gradOutputTr(gradOutput);
    DiopiTensor indicesTr(indices);

    DIOPI_CHECK(inputTr.dim() == 3 || inputTr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    if (inputTr.dim() == 3) {
        inputTr.unsqueeze(0);
        indicesTr.unsqueeze(0);
        gradInputTr.unsqueeze(0);
        gradOutputTr.unsqueeze(0);
    }

    std::vector<DiopiTensor*> pTensors{&inputTr, &gradOutputTr};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));

    std::vector<DiopiTensor*> pTensorsIndices{&indicesTr};
    DIOPI_CALL(autoCastTensorType(ctx, pTensorsIndices, {diopi_dtype_int16, diopi_dtype_int32}));

    diopiMemoryFormat_t memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    DIOPI_CALL(contiguous(ctx, inputTr, memoryFormat));
    DIOPI_CALL(contiguous(ctx, gradOutputTr, memoryFormat));
    DIOPI_CALL(contiguous(ctx, indicesTr, memoryFormat));
    DiopiTensor gradInputTmpTr = requiresTensor(ctx, gradInputTr.shape(), inputTr.dtype(), memoryFormat);

    std::vector<int64_t> inputDim = inputTr.shape();
    std::vector<int64_t> gradInputDim = gradInputTmpTr.shape();
    std::vector<int64_t> gradOutputDim = gradOutputTr.shape();
    std::vector<int64_t> indicesDim = indicesTr.shape();
    CnnlTensorDesc inputDesc(inputTr, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc gradInputDesc(gradInputTmpTr, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc gradOutputDesc(gradOutputTr, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc indicesDesc(indicesTr, CNNL_LAYOUT_NHWC);

    auto[kernelH, kernelW] = extractDims(kernelSize);
    auto[strideH, strideW] = stride.len == 0 ? std::make_pair(kernelH, kernelW) : extractDims(stride);
    auto[padH, padW] = extractDims(padding);
    auto[dilation0, dilation1] = extractDims(dilation);

    // calculate padding coefficients
    auto padLeft = padW, padRight = padW, padUp = padH, padDown = padH;
    int height = (gradOutputDim[2] - 1) * strideH + kernelH;
    int width = (gradOutputDim[3] - 1) * strideW + kernelW;
    if (padH + inputDim[2] >= height) {
        padDown = 0;
    }
    if (padW + inputDim[3] >= width) {
        padRight = 0;
    }
    // if ceil_mode is set to true, the pad needs to be filled up.
    if (ceilMode) {
        padDown = height - inputDim[2] - padH;
        padRight = width - inputDim[3] - padW;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> cnnlPoolDesc;
    cnnlPoolingDescriptor_t poolDesc = cnnlPoolDesc.get();
    DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
        poolDesc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernelH, kernelW, padUp, padDown, padLeft, padRight, strideH, strideW, dilation0, dilation1, ceilMode));

    DIOPI_CALLCNNL(cnnlPoolingBackward(handle,
                                       poolDesc,
                                       nullptr,
                                       indicesDesc.get(),
                                       indicesTr.data(),
                                       gradOutputDesc.get(),
                                       gradOutputTr.data(),
                                       inputDesc.get(),
                                       inputTr.data(),
                                       nullptr,
                                       gradInputDesc.get(),
                                       gradInputTmpTr.data()));

    // Channels last -> contiguous
    DIOPI_CALL(contiguous(ctx, gradInputTmpTr, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(diopiCopyInp(ctx, gradInputTmpTr.tensorHandle(), gradInputTr.tensorHandle()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
