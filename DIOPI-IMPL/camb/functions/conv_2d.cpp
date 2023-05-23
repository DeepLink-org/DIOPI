/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {
diopiError_t tensorPermute(diopiContextHandle_t ctx, DiopiTensor &dstTensor, DiopiTensor srcTensor, std::vector<int64_t> permAxis) {
    if (!dstTensor.defined()) {
        std::vector<int64_t> srcShapeT64(srcTensor.shape().size());
        for (int i = 0; i < srcTensor.shape().size(); ++i) {
            srcShapeT64[i] = srcTensor.shape()[permAxis[i]];
        }
        diopiSize_t srcTShape(srcShapeT64.data(), srcShapeT64.size());
        auto dstHandle = dstTensor.tensorHandle();
        DIOPI_CALL(diopiRequireTensor(ctx, &dstHandle, &srcTShape, nullptr, srcTensor.dtype(), diopi_device));
        dstTensor = DiopiTensor(dstHandle);
    }
    diopiSize_t axisSize(permAxis.data(), 4);
    DIOPI_CALL(diopiPermute(ctx, dstTensor.tensorHandle(), srcTensor.tensorHandle(), axisSize));
    return diopiSuccess;
}

diopiError_t tensorPermute2D(diopiContextHandle_t ctx, DiopiTensor &dst, DiopiTensor src, MemoryFormat format) {
    if (src.isContiguous(format)) {
        dst = src;
        return diopiSuccess;
    }
    if (src.isContiguous(MemoryFormat::Contiguous) && format == MemoryFormat::ChannelsLast) {
        DIOPI_CALL(tensorPermute(ctx, dst, src, {0, 2, 3, 1}));
        return diopiSuccess;
    }
    if (src.isContiguous(MemoryFormat::ChannelsLast) && format == MemoryFormat::Contiguous) {
        DIOPI_CALL(tensorPermute(ctx, dst, src, {0, 3, 1, 2}))
    }
    return diopiErrorOccurred;
}

}  // namespace

extern "C" diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor outputTensor(out);

    DIOPI_CHECK(inputTensor.isContiguous() || inputTensor.isContiguous(MemoryFormat::ChannelsLast), "%s",
                "[diopiConvolution2d] the memory format is not supportted.");
    DIOPI_CHECK(weightTensor.isContiguous() || weightTensor.isContiguous(MemoryFormat::ChannelsLast), "%s",
                "[diopiConvolution2d] the memory format is not supportted.");
    DIOPI_CHECK(outputTensor.isContiguous() || outputTensor.isContiguous(MemoryFormat::ChannelsLast), "%s",
                "[diopiConvolution2d] the memory format is not supportted.");

    DiopiTensor inputTensorCasted = inputTensor;
    DiopiTensor weightTensorCasted = weightTensor;
    DiopiTensor outputTensorCasted = outputTensor;

    std::vector<DiopiTensor *> tensors{&inputTensorCasted, &weightTensorCasted, &outputTensorCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    DiopiTensor inputTensorT, weightTensorT, outputTensorT;

    DIOPI_CALL(tensorPermute2D(ctx, inputTensorT, inputTensorCasted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(tensorPermute2D(ctx, outputTensorT, outputTensorCasted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(tensorPermute2D(ctx, weightTensorT, weightTensorCasted, MemoryFormat::ChannelsLast));

    std::vector<int32_t> inputTShape{inputTensorT.shape().begin(), inputTensorT.shape().end()};
    std::vector<int32_t> weightTShape{weightTensorT.shape().begin(), weightTensorT.shape().end()};
    std::vector<int32_t> outputTShape{outputTensorT.shape().begin(), outputTensorT.shape().end()};

    CnnlTensorDesc inputDesc(inputTensorT, CNNL_LAYOUT_NHWC, inputTShape);
    CnnlTensorDesc weightDesc(weightTensorT, CNNL_LAYOUT_NHWC, weightTShape);
    CnnlTensorDesc outputDesc(outputTensorT, CNNL_LAYOUT_NHWC, outputTShape);

    DiopiTensor biasTensor(bias);
    DiopiTensor biasTensorCasted = biasTensor;
    CnnlTensorDesc biasDesc;
    if (nullptr != bias) {
        std::vector<DiopiTensor *> tensors{&biasTensorCasted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        DIOPI_CALL(biasDesc.set(biasTensorCasted, CNNL_LAYOUT_ARRAY));
    }

    std::vector<int> strideVec{stride.data, stride.data + stride.len};
    std::vector<int> paddingVec{padding.data, padding.data + padding.len};
    std::vector<int> dilationVec{dilation.data, dilation.data + dilation.len};

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    int paddingTmp[4] = {paddingVec[0], paddingVec[0], paddingVec[1], paddingVec[1]};
    int strideTmp[2] = {strideVec[0], strideVec[1]};
    int dilationTmp[2] = {dilationVec[0], dilationVec[1]};

    cnnlDataType_t computeType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, inputTensorT.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc.get(), 4, paddingTmp, strideTmp, dilationTmp, groups, computeType));

    size_t workspaceSize;
    DIOPI_CALLCNNL(cnnlGetConvolutionForwardWorkspaceSize(
        handle, inputDesc.get(), weightDesc.get(), outputDesc.get(), biasDesc.get(), convDesc.get(), CNNL_CONVOLUTION_FWD_ALGO_DIRECT, &workspaceSize));

    void *workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlConvolutionForward(handle,
                                          convDesc.get(),
                                          CNNL_CONVOLUTION_FWD_ALGO_DIRECT,
                                          nullptr,
                                          inputDesc.get(),
                                          inputTensorT.data(),
                                          weightDesc.get(),
                                          weightTensorT.data(),
                                          biasTensor.defined() ? biasDesc.get() : nullptr,
                                          biasTensor.defined() ? biasTensorCasted.data() : nullptr,
                                          workspace,
                                          workspaceSize,
                                          nullptr,
                                          outputDesc.get(),
                                          outputTensorT.data()));

    DIOPI_CALL(tensorPermute2D(ctx, outputTensorCasted, outputTensorCasted, MemoryFormat::Contiguous));
    DIOPI_CALL(dataTypeCast(ctx, outputTensor, outputTensorCasted));
    return diopiSuccess;
}

extern "C" diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                                   diopiTensorHandle_t grad3, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                                   diopiConstTensorHandle_t weight, diopiSize_t *biasSizes, diopiSize_t stride, diopiSize_t padding,
                                                   diopiSize_t dilation, bool transposed, diopiSize_t outputPadding, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradWeightTensor(gradWeight);

    DIOPI_CHECK(inputTensor.isContiguous() || inputTensor.isContiguous(MemoryFormat::ChannelsLast), "%s",
                "[diopiConvolution2dBackward] the memory format is not supportted.");
    DIOPI_CHECK(weightTensor.isContiguous() || weightTensor.isContiguous(MemoryFormat::ChannelsLast), "%s",
                "[diopiConvolution2dBackward] the memory format is not supportted.");
    DIOPI_CHECK(gradOutputTensor.isContiguous() || gradOutputTensor.isContiguous(MemoryFormat::ChannelsLast), "%s",
                "[diopiConvolution2dBackward] the memory format is not supportted.");
    DIOPI_CHECK(gradInputTensor.isContiguous() || gradInputTensor.isContiguous(MemoryFormat::ChannelsLast), "%s",
                "[diopiConvolution2dBackward] the memory format is not supportted.");
    DIOPI_CHECK(gradWeightTensor.isContiguous() || gradWeightTensor.isContiguous(MemoryFormat::ChannelsLast), "%s",
                "[diopiConvolution2dBackward] the memory format is not supportted.");

    DiopiTensor inputCasted = inputTensor;
    DiopiTensor weightCasted = weightTensor;
    DiopiTensor gradOutputCasted = gradOutputTensor;
    DiopiTensor gradInputCasted = gradInputTensor;
    DiopiTensor gradWeightCasted = gradWeightTensor;

    std::vector<DiopiTensor *> tensors{&inputCasted, &weightCasted, &gradOutputCasted, &gradInputCasted, &gradWeightCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    DiopiTensor inputT, weightT, gradOutputT, gradInputT, gradWeightT;

    DIOPI_CALL(tensorPermute2D(ctx, inputT, inputCasted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(tensorPermute2D(ctx, weightT, weightCasted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(tensorPermute2D(ctx, gradInputT, gradInputCasted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(tensorPermute2D(ctx, gradOutputT, gradOutputCasted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(tensorPermute2D(ctx, gradWeightT, gradWeightCasted, MemoryFormat::ChannelsLast));

    std::vector<int32_t> inputTShape{inputT.shape().begin(), inputT.shape().end()};
    std::vector<int32_t> weightTShape{weightT.shape().begin(), weightT.shape().end()};
    std::vector<int32_t> gradOutputTShape{gradOutputT.shape().begin(), gradOutputT.shape().end()};
    std::vector<int32_t> gradInputTShape{gradInputT.shape().begin(), gradInputT.shape().end()};
    std::vector<int32_t> gradWeightShape{gradWeightT.shape().begin(), gradWeightT.shape().end()};

    CnnlTensorDesc inputDesc(inputT, CNNL_LAYOUT_NHWC, inputTShape);
    CnnlTensorDesc weightDesc(weightT, CNNL_LAYOUT_NHWC, weightTShape);
    CnnlTensorDesc outputGradDesc(gradOutputT, CNNL_LAYOUT_NHWC, gradOutputTShape);
    CnnlTensorDesc inputGradDesc(gradInputT, CNNL_LAYOUT_NHWC, gradInputTShape);
    CnnlTensorDesc weightGradDesc(gradWeightT, CNNL_LAYOUT_NHWC, gradWeightShape);

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    std::vector<int> strideVec{stride.data, stride.data + stride.len};
    std::vector<int> paddingVec{padding.data, padding.data + padding.len};
    std::vector<int> dilationVec{dilation.data, dilation.data + dilation.len};

    int paddingTmp[4] = {paddingVec[0], paddingVec[1], paddingVec[0], paddingVec[1]};
    int strideTmp[2] = {strideVec[0], strideVec[1]};
    int dilationTmp[2] = {dilationVec[0], dilationVec[1]};

    cnnlDataType_t computeType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, inputT.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc.get(), 4, paddingTmp, strideTmp, dilationTmp, groups, computeType));

    size_t workspaceSizeFilter = 0;
    DIOPI_CALLCNNL(cnnlGetConvolutionBackwardFilterWorkspaceSize(
        handle, inputDesc.get(), outputGradDesc.get(), weightDesc.get(), convDesc.get(), CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT, &workspaceSizeFilter));

    void *workspaceFilter = nullptr;
    if (workspaceSizeFilter != 0) {
        workspaceFilter = requiresBuffer(ctx, workspaceSizeFilter).data();
    }

    DIOPI_CALLCNNL(cnnlConvolutionBackwardFilter(handle,
                                                 nullptr,
                                                 inputDesc.get(),
                                                 inputT.data(),
                                                 outputGradDesc.get(),
                                                 gradOutputT.data(),
                                                 convDesc.get(),
                                                 CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT,
                                                 workspaceFilter,
                                                 workspaceSizeFilter,
                                                 nullptr,
                                                 weightGradDesc.get(),
                                                 gradWeightT.data()));

    size_t workspaceSizeInput;
    DIOPI_CALLCNNL(cnnlGetConvolutionBackwardDataWorkspaceSize(handle,
                                                               weightDesc.get(),
                                                               outputGradDesc.get(),
                                                               convDesc.get(),
                                                               inputGradDesc.get(),
                                                               CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                                               &workspaceSizeInput));

    void *workspaceInput;
    if (workspaceSizeInput != 0) {
        workspaceInput = requiresBuffer(ctx, workspaceSizeInput).data();
    }

    DIOPI_CALLCNNL(cnnlConvolutionBackwardData(handle,
                                               nullptr,
                                               weightDesc.get(),
                                               weightT.data(),
                                               outputGradDesc.get(),
                                               gradOutputT.data(),
                                               convDesc.get(),
                                               CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                               workspaceInput,
                                               workspaceSizeInput,
                                               nullptr,
                                               inputGradDesc.get(),
                                               gradInputT.data()));

    DIOPI_CALL(tensorPermute2D(ctx, gradInputCasted, gradInputT, MemoryFormat::Contiguous));
    DIOPI_CALL(tensorPermute2D(ctx, gradWeightCasted, gradWeightT, MemoryFormat::Contiguous));

    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputCasted));
    DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, gradWeightCasted));

    if (grad3 != nullptr) {
        DiopiTensor biasGradTensor(grad3);
        DiopiTensor gradBiasCasted = biasGradTensor;
        std::vector<DiopiTensor *> tensors{&gradBiasCasted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        CnnlTensorDesc biasGradDesc(gradBiasCasted, CNNL_LAYOUT_ARRAY);
        std::vector<int64_t> biasShape{biasGradTensor.shape().begin(), biasGradTensor.shape().end()};
        biasSizes->data = biasShape.data();
        biasSizes->len = biasShape.size();
        size_t workspaceSizeBias;
        DIOPI_CALLCNNL(cnnlGetBiasAddBackwardWorkspaceSize(handle, outputGradDesc.get(), biasGradDesc.get(), 3, &workspaceSizeBias))
        void *workspaceBias = nullptr;
        if (0 != workspaceSizeBias) {
            workspaceBias = requiresBuffer(ctx, workspaceSizeBias).data();
        }
        DIOPI_CALLCNNL(cnnlBiasAddBackward_v2(
            handle, outputGradDesc.get(), gradOutputT.data(), 3, biasGradDesc.get(), gradBiasCasted.data(), workspaceBias, workspaceSizeBias));
        DIOPI_CALL(dataTypeCast(ctx, biasGradTensor, gradBiasCasted))
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
