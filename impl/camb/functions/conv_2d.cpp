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

#define REQUIRES_TENSOR_BY_DTYPE_OR_NOT(tensor1, tensor2, targetDtype)                           \
    DiopiTensor tensor1 = tensor2;                                                               \
    if (tensor2.defined() && tensor1.dtype() != targetDtype) {                                   \
        tensor1 = requiresTensor(ctx, tensor1.shape(), targetDtype, MemoryFormat::ChannelsLast); \
    }
namespace {
diopiError_t convForward(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor weight, DiopiTensor bias, DiopiTensor output, diopiSize_t stride,
                         diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weightDesc(weight, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc outputDesc(output, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc biasDesc;
    if (bias.defined()) {
        DIOPI_CALL(biasDesc.set(bias, CNNL_LAYOUT_NHWC));
    }

    std::vector<int> strideVec{stride.data, stride.data + stride.len};
    std::vector<int> paddingVec{padding.data, padding.data + padding.len};
    std::vector<int> dilationVec{dilation.data, dilation.data + dilation.len};

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    int paddingTmp[4] = {paddingVec[0], paddingVec[0], paddingVec[1], paddingVec[1]};
    int strideTmp[2] = {strideVec[0], strideVec[1]};
    int dilationTmp[2] = {dilationVec[0], dilationVec[1]};

    cnnlDataType_t computeType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, input.dtype()));
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
                                          input.data(),
                                          weightDesc.get(),
                                          weight.data(),
                                          bias.defined() ? biasDesc.get() : nullptr,
                                          bias.defined() ? bias.data() : nullptr,
                                          workspace,
                                          workspaceSize,
                                          nullptr,
                                          outputDesc.get(),
                                          output.data()));
    return diopiSuccess;
}

diopiError_t convBackwardData(diopiContextHandle_t ctx, DiopiTensor gradOutput, DiopiTensor gradInput, DiopiTensor weight, diopiSize_t stride,
                              diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlTensorDesc gradOutputDesc(gradOutput, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc gradInputDesc(gradInput, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weightDesc(weight, CNNL_LAYOUT_NHWC);

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    std::vector<int> strideVec{stride.data, stride.data + stride.len};
    std::vector<int> paddingVec{padding.data, padding.data + padding.len};
    std::vector<int> dilationVec{dilation.data, dilation.data + dilation.len};

    int paddingTmp[4] = {paddingVec[0], paddingVec[0], paddingVec[1], paddingVec[1]};
    int strideTmp[2] = {strideVec[0], strideVec[1]};
    int dilationTmp[2] = {dilationVec[0], dilationVec[1]};

    cnnlDataType_t computeType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, gradInput.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc.get(), 4, paddingTmp, strideTmp, dilationTmp, groups, computeType));

    size_t workspaceSize;
    DIOPI_CALLCNNL(cnnlGetConvolutionBackwardDataWorkspaceSize(
        handle, weightDesc.get(), gradOutputDesc.get(), convDesc.get(), gradInputDesc.get(), CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT, &workspaceSize));

    void *workspace;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlConvolutionBackwardData(handle,
                                               nullptr,
                                               weightDesc.get(),
                                               weight.data(),
                                               gradOutputDesc.get(),
                                               gradOutput.data(),
                                               convDesc.get(),
                                               CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                               workspace,
                                               workspaceSize,
                                               nullptr,
                                               gradInputDesc.get(),
                                               gradInput.data()));

    return diopiSuccess;
}

diopiError_t convBackwardFilter(diopiContextHandle_t ctx, DiopiTensor gradOutput, DiopiTensor gradWeight, DiopiTensor input, diopiSize_t stride,
                                diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc gradOutputDesc(gradOutput, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc gradWeightDesc(gradWeight, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_NHWC);

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    std::vector<int> strideVec{stride.data, stride.data + stride.len};
    std::vector<int> paddingVec{padding.data, padding.data + padding.len};
    std::vector<int> dilationVec{dilation.data, dilation.data + dilation.len};

    int paddingTmp[4] = {paddingVec[0], paddingVec[0], paddingVec[1], paddingVec[1]};
    int strideTmp[2] = {strideVec[0], strideVec[1]};
    int dilationTmp[2] = {dilationVec[0], dilationVec[1]};

    cnnlDataType_t computeType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, gradOutput.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc.get(), 4, paddingTmp, strideTmp, dilationTmp, groups, computeType));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetConvolutionBackwardFilterWorkspaceSize(
        handle, inputDesc.get(), gradOutputDesc.get(), gradWeightDesc.get(), convDesc.get(), CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT, &workspaceSize));

    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlConvolutionBackwardFilter(handle,
                                                 nullptr,
                                                 inputDesc.get(),
                                                 input.data(),
                                                 gradOutputDesc.get(),
                                                 gradOutput.data(),
                                                 convDesc.get(),
                                                 CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT,
                                                 workspace,
                                                 workspaceSize,
                                                 nullptr,
                                                 gradWeightDesc.get(),
                                                 gradWeight.data()));
    return diopiSuccess;
}

diopiError_t convBackwardBias(diopiContextHandle_t ctx, DiopiTensor gradOutput, DiopiTensor gradBias) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc gradOutputDesc(gradOutput, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc biasGradDesc(gradBias, CNNL_LAYOUT_NHWC);
    std::vector<int64_t> biasShape = gradBias.shape();
    size_t workspaceSizeBias;
    DIOPI_CALLCNNL(cnnlGetBiasAddBackwardWorkspaceSize(handle, gradOutputDesc.get(), biasGradDesc.get(), 3, &workspaceSizeBias))
    void *workspaceBias = nullptr;
    if (0 != workspaceSizeBias) {
        workspaceBias = requiresBuffer(ctx, workspaceSizeBias).data();
    }
    DIOPI_CALLCNNL(
        cnnlBiasAddBackward_v2(handle, gradOutputDesc.get(), gradOutput.data(), 3, biasGradDesc.get(), gradBias.data(), workspaceBias, workspaceSizeBias));
    return diopiSuccess;
}

}  // namespace

extern "C" diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor outputTensor(out);
    DiopiTensor biasTensor(bias);

    DIOPI_CHECK(inputTensor.isContiguous(MemoryFormat::ChannelsLast), "inputTensor should be ChannelsLast");
    DIOPI_CHECK(weightTensor.isContiguous(MemoryFormat::ChannelsLast), "weightTensor should be ChannelsLast");
    DIOPI_CHECK(outputTensor.isContiguous(MemoryFormat::ChannelsLast), "outputTensor should be ChannelsLast");

    std::vector<DiopiTensor *> tensors{&inputTensor, &weightTensor};
    if (bias) {
        tensors.push_back(&biasTensor);
    }
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(outputTensorTmp, outputTensor, inputTensor.dtype());

    DIOPI_CALL(convForward(ctx, inputTensor, weightTensor, biasTensor, outputTensorTmp, stride, padding, dilation, groups));
    DIOPI_CALL(dataTypeCast(ctx, outputTensor, outputTensorTmp));
    return diopiSuccess;
}

extern "C" diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                                   diopiTensorHandle_t grad3, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                                   diopiConstTensorHandle_t weight, diopiSize_t *biasSizes, diopiSize_t stride, diopiSize_t padding,
                                                   diopiSize_t dilation, bool transposed, diopiSize_t outputPadding, int64_t groups) {
    if (!gradInput && !gradWeight && !grad3) {
        // do nothing
        return diopiSuccess;
    }
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradWeightTensor(gradWeight);
    DiopiTensor gradBiasTensor(grad3);

    DIOPI_CHECK(inputTensor.isContiguous(MemoryFormat::ChannelsLast), "inputTensor should be ChannelsLast");
    if (gradInputTensor.defined()) {
        DIOPI_CHECK(gradInputTensor.isContiguous(MemoryFormat::ChannelsLast), "gradInputTensor should be ChannelsLast");
    }
    std::vector<DiopiTensor *> tensors{&inputTensor, &weightTensor, &gradOutputTensor};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    if (gradWeightTensor.defined()) {
        REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradWeightTensorTmp, gradWeightTensor, inputTensor.dtype());
        DIOPI_CALL(convBackwardFilter(ctx, gradOutputTensor, gradWeightTensorTmp, inputTensor, stride, padding, dilation, groups));
        DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, gradWeightTensorTmp));
    }

    if (gradInputTensor.defined()) {
        REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradInputTensorTmp, gradInputTensor, inputTensor.dtype());
        DIOPI_CALL(convBackwardData(ctx, gradOutputTensor, gradInputTensorTmp, weightTensor, stride, padding, dilation, groups));
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));
    }

    if (grad3 != nullptr) {
        REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradBiasTensorTmp, gradBiasTensor, inputTensor.dtype());
        DIOPI_CALL(convBackwardBias(ctx, gradOutputTensor, gradBiasTensorTmp));
        DIOPI_CALL(dataTypeCast(ctx, gradBiasTensor, gradBiasTensorTmp))
    }

    return diopiSuccess;
}

extern "C" diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                             diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t outputPadding, int64_t groups,
                                             diopiSize_t dilation) {
    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor biasTensor(bias);

    DiopiTensor inputCasted = inputTensor;
    DiopiTensor weightCasted = weightTensor;
    DiopiTensor outputCasted = outTensor;

    std::vector<DiopiTensor *> tensors{&inputCasted, &weightCasted, &outputCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DIOPI_CALL(convBackwardData(ctx, inputCasted, outputCasted, weightCasted, stride, padding, dilation, groups));

    if (biasTensor.defined()) {
        cnnlHandle_t handle = cnnlHandlePool.get(ctx);
        DiopiTensor biasCasted = biasTensor;
        std::vector<DiopiTensor *> tensors{&biasCasted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        CnnlTensorDesc biasDesc(biasCasted, CNNL_LAYOUT_NHWC);

        CnnlTensorDesc outputDesc(outputCasted, CNNL_LAYOUT_NHWC);
        size_t workspaceSizeBias;
        DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, biasDesc.get(), outputDesc.get(), &workspaceSizeBias));

        void *workspaceBias = nullptr;
        if (workspaceSizeBias != 0) {
            workspaceBias = requiresBuffer(ctx, workspaceSizeBias).data();
        }
        float alpha = 1.0;
        float beta = 1.0;
        DIOPI_CALLCNNL(
            cnnlBiasAdd(handle, &alpha, biasDesc.get(), biasCasted.data(), workspaceBias, workspaceSizeBias, &beta, outputDesc.get(), outputCasted.data()));
    }

    DIOPI_CALL(dataTypeCast(ctx, outTensor, outputCasted));
    return diopiSuccess;
}

extern "C" diopiError_t diopiConvTranspose2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                                     diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                                     diopiConstTensorHandle_t weight, diopiSize_t *biasSizes, diopiSize_t stride, diopiSize_t padding,
                                                     diopiSize_t dilation, diopiSize_t outputPadding, int64_t groups) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradWeightTensor(gradWeight);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor gradBiasTensor(gradBias);

    DiopiTensor gradInputCasted = gradInputTensor;
    DiopiTensor gradWeightCasted = gradWeightTensor;
    DiopiTensor gradOutputCasted = gradOutputTensor;
    DiopiTensor inputCasted = inputTensor;
    DiopiTensor weightCasted = weightTensor;

    std::vector<DiopiTensor *> tensors{&gradInputCasted, &gradWeightCasted, &gradOutputCasted, &inputCasted, &weightCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DIOPI_CALL(convForward(ctx, gradOutputCasted, weightCasted, {}, gradInputCasted, stride, padding, dilation, groups));
    DIOPI_CALL(convBackwardFilter(ctx, inputCasted, gradWeightCasted, gradOutputCasted, stride, padding, dilation, groups));
    if (gradBiasTensor.defined()) {
        DiopiTensor gradBiasCasted = gradBiasTensor;
        std::vector<DiopiTensor *> tensors{&gradBiasCasted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        DIOPI_CALL(convBackwardBias(ctx, gradOutputCasted, gradBiasCasted));
        DIOPI_CALL(dataTypeCast(ctx, gradBiasTensor, gradBiasCasted))
    }

    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputCasted));
    DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, gradWeightCasted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
