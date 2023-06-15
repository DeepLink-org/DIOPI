/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstddef>
#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../diopi_helper.hpp"

namespace impl {

namespace camb {

// Integer division rounding to -Infinity
template <typename T>
static inline T divRtn(T x, T y) {
    int q = x / y;
    int r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) --q;
    return q;
}

static diopiError_t im2colShapeCheck(const DiopiTensor& input, int kernelHeight, int kernelWidth, int dilationHeight, int dilationWidth, int padHeight,
                                     int padWidth, int strideHeight, int strideWidth) {
    DIOPI_CHECK(
        kernelWidth > 0 && kernelHeight > 0, "kernel size should be greater than zero, but got kernelHeight:%d kernelWidth:%d", kernelHeight, kernelWidth);

    DIOPI_CHECK(dilationWidth > 0 && dilationHeight > 0,
                "dilation should be greater than zero, but got dilationHeight:%d dilationWidth:%d",
                dilationHeight,
                dilationWidth);

    DIOPI_CHECK(padWidth >= 0 && padHeight >= 0, "padding should be non-negative, but got padHeight:%d padWidth:%d", padHeight, padWidth);

    DIOPI_CHECK(strideWidth > 0 && strideHeight > 0, "stride should be greater than zero, but got strideHeight:%d strideWidth:%d", strideHeight, strideWidth);

    int64_t ndim = input.dim();

    DIOPI_CHECK(input.numel() != 0 && (ndim == 3 || ndim == 4), "Expected non-empty 3D or 4D input tensor, but the dim of input:%d", input.dim());

    int64_t dimBatch = 0;

    if (ndim == 3) {
        dimBatch = -1;
    }

    int64_t inputHeight = input.size(dimBatch + 2);
    int64_t inputWidth = input.size(dimBatch + 3);
    int64_t outputHeight = divRtn<int64_t>(inputHeight + 2 * static_cast<int64_t>(padHeight) - (dilationHeight * (kernelHeight - 1) + 1), strideHeight) + 1;
    int64_t outputWidth = divRtn<int64_t>(inputWidth + 2 * static_cast<int64_t>(padWidth) - (dilationWidth * (kernelWidth - 1) + 1), strideWidth) + 1;

    DIOPI_CHECK(outputHeight >= 1 && outputWidth >= 1,
                "Given input with spatial size (%d, %d), kernel_size=(%d, %d), dilation=(%d, %d), padding=(%d,%d), calculated shape of the array of sliding"
                "blocks as(% d, % d) which is too small(non - positive).",
                inputHeight,
                inputHeight,
                kernelHeight,
                kernelWidth,
                dilationHeight,
                dilationWidth,
                padHeight,
                padWidth,
                outputHeight,
                outputWidth);
    return diopiSuccess;
}

diopiError_t im2colOutInternal(diopiContextHandle_t ctx, DiopiTensor& output, const DiopiTensor& input, const std::vector<int>& kernelSize,
                               const std::vector<int>& dilation, const std::vector<int>& padding, const std::vector<int>& stride) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(output);

    CnnlTensorDesc descInput(inputTr, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc descOutput(outputTr, CNNL_LAYOUT_NCHW);
    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    // Just for pass filter kernel info, so fake input, stride, and dtype.
    const int ci = input.size(1);
    std::vector<int> weightSizes{ci, ci, kernelSize[0], kernelSize[1]};
    std::vector<int> weightStrides{1, 1, 1, 1};
    CnnlTensorDesc descWeight;
    descWeight.set(outputTr.dtype(), weightSizes, weightStrides, CNNL_LAYOUT_NCHW);

    descOutput.set(output, CNNL_LAYOUT_ARRAY);
    const int64_t groups = 1;
    cnnlDataType_t computeType;
    std::vector<int32_t> paddingTmp{padding[0], padding[0], padding[1], padding[1]};
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, outputTr.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc.get(), inputTr.dim(), paddingTmp.data(), stride.data(), dilation.data(), groups, computeType));

    auto inputPtr = inputTr.data();
    auto outputPtr = outputTr.data();
    void* workspacePtr = nullptr;
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetIm2ColWorkspaceSize(handle, descInput.get(), descWeight.get(), convDesc.get(), descOutput.get(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlIm2Col(
        handle, descInput.get(), inputPtr, descWeight.get(), convDesc.get(), nullptr, nullptr, nullptr, workspace, workspaceSize, descOutput.get(), outputPtr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernelSize,
                                    diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTr(input);
    DiopiTensor outTr(out);
    DIOPI_CHECK(kernelSize.len == 2, "The length of the kernelSize's shape should equal to 2, but got len: %d", kernelSize.len);
    DIOPI_CHECK(dilation.len == 2, "The length of the dilation's shape should equal to 2, but got len: %d", dilation.len);
    DIOPI_CHECK(padding.len == 2, "The length of the padding's shape should equal to 2, but got len: %d", padding.len);
    DIOPI_CHECK(stride.len == 2, "The length of the stride's shape should equal to 2, but got len: %d", stride.len);

    std::vector<int32_t> kernelVec = diopiSizeT2Vector<int32_t>(kernelSize);
    std::vector<int32_t> dilationVec = diopiSizeT2Vector<int32_t>(dilation);
    std::vector<int32_t> paddingVec = diopiSizeT2Vector<int32_t>(padding);
    std::vector<int32_t> strideVec = diopiSizeT2Vector<int32_t>(stride);

    int32_t kernelHeight = kernelVec[0];
    int32_t kernelWidth = kernelVec[1];
    int32_t dilationHeight = dilationVec[0];
    int32_t dilationWidth = dilationVec[1];
    int32_t padHeight = paddingVec[0];
    int32_t padWidth = paddingVec[1];
    int32_t strideHeight = strideVec[0];
    int32_t strideWidth = strideVec[1];
    DIOPI_CALL(im2colShapeCheck(inputTr, kernelHeight, kernelWidth, dilationHeight, dilationWidth, padHeight, padWidth, strideHeight, strideWidth));

    if (inputTr.dim() == 3) {
        inputTr.unsqueeze(0);
    }

    DIOPI_CALL(contiguous(ctx, inputTr));

    int64_t batchSize = inputTr.size(0);
    int64_t nInputPlane = inputTr.size(1);
    int64_t inputHeight = inputTr.size(2);
    int64_t inputWidth = inputTr.size(3);

    int64_t outputHeight = (inputHeight + 2 * static_cast<int64_t>(padHeight) - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1;
    int64_t outputWidth = (inputWidth + 2 * static_cast<int64_t>(padWidth) - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1;
    int64_t nOutputPlane = nInputPlane * kernelWidth * kernelHeight;
    int64_t outputLength = outputHeight * outputWidth;

    outTr.reshape({batchSize, outputLength, nOutputPlane});
    DiopiTensor outputTr;
    outputTr = requiresTensor(ctx, outTr.shape(), outTr.dtype());
    DIOPI_CALL(im2colOutInternal(ctx, outputTr, inputTr, kernelVec, dilationVec, paddingVec, strideVec));
    //  CNNL kernel output shape is different with original tensor, so need to transpose
    DIOPI_CALL(transpose(ctx, outTr, outputTr, 1, 2));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl