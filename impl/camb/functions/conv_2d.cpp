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
diopiError_t convForward(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor weight, DiopiTensor bias, DiopiTensor output, diopiSize_t stride,
                         diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_NHWC, input.shape<int32_t>());
    CnnlTensorDesc weightDesc(weight, CNNL_LAYOUT_NHWC, weight.shape<int32_t>());
    CnnlTensorDesc outputDesc(output, CNNL_LAYOUT_NHWC, output.shape<int32_t>());
    CnnlTensorDesc biasDesc;
    if (bias.defined()) {
        DIOPI_CALL(biasDesc.set(bias, CNNL_LAYOUT_ARRAY));
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

    CnnlTensorDesc gradOutputDesc(gradOutput, CNNL_LAYOUT_NHWC, gradOutput.shape<int32_t>());
    CnnlTensorDesc gradInputDesc(gradInput, CNNL_LAYOUT_NHWC, gradInput.shape<int32_t>());
    CnnlTensorDesc weightDesc(weight, CNNL_LAYOUT_NHWC, weight.shape<int32_t>());

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
    CnnlTensorDesc gradOutputDesc(gradOutput, CNNL_LAYOUT_NHWC, gradOutput.shape<int32_t>());
    CnnlTensorDesc gradWeightDesc(gradWeight, CNNL_LAYOUT_NHWC, gradWeight.shape<int32_t>());
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_NHWC, input.shape<int32_t>());

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
    CnnlTensorDesc gradOutputDesc(gradOutput, CNNL_LAYOUT_NHWC, gradOutput.shape<int32_t>());
    CnnlTensorDesc biasGradDesc(gradBias, CNNL_LAYOUT_ARRAY);
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
    DiopiTensor biasTensor(bias);
    DiopiTensor outputTensor(out);

    DiopiTensor inputTensorCasted = inputTensor;
    DiopiTensor weightTensorCasted = weightTensor;
    DiopiTensor biasTensorCasted = biasTensor;
    DiopiTensor outputTensorCasted = outputTensor;

    std::vector<DiopiTensor *> tensors{&inputTensorCasted, &weightTensorCasted, &outputTensorCasted};
    if (biasTensorCasted.defined()) {
        tensors.push_back(&biasTensorCasted);
    }
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    DiopiTensor inputTensorTr, weightTensorTr, outputTensorTr;
    DIOPI_CALL(transpose(ctx, inputTensorCasted, inputTensorTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, weightTensorCasted, weightTensorTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, outputTensorCasted, outputTensorTr, {0, 2, 3, 1}));

    DIOPI_CALL(convForward(ctx, inputTensorTr, weightTensorTr, biasTensorCasted, outputTensorTr, stride, padding, dilation, groups));
    DIOPI_CALL(transpose(ctx, outputTensorTr, outputTensorCasted, {0, 3, 1, 2}));
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

    DiopiTensor inputCasted = inputTensor;
    DiopiTensor weightCasted = weightTensor;
    DiopiTensor gradOutputCasted = gradOutputTensor;
    DiopiTensor gradInputCasted = gradInputTensor;
    DiopiTensor gradWeightCasted = gradWeightTensor;

    std::vector<DiopiTensor *> tensors{&inputCasted, &weightCasted, &gradOutputCasted, &gradInputCasted, &gradWeightCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    DiopiTensor inputTr, weightTr, gradOutputTr, gradInputTr, gradWeightTr;
    DIOPI_CALL(transpose(ctx, inputCasted, inputTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, weightCasted, weightTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, gradOutputCasted, gradOutputTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, gradInputCasted, gradInputTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, gradWeightCasted, gradWeightTr, {0, 2, 3, 1}));

    DIOPI_CALL(convBackwardData(ctx, gradOutputTr, gradInputTr, weightTr, stride, padding, dilation, groups));
    DIOPI_CALL(convBackwardFilter(ctx, gradOutputTr, gradWeightTr, inputTr, stride, padding, dilation, groups));
    DiopiTensor biasGradTensor(grad3);
    if (biasGradTensor.defined()) {
        DiopiTensor gradBiasCasted = biasGradTensor;
        std::vector<DiopiTensor *> tensors{&gradBiasCasted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        DIOPI_CALL(convBackwardBias(ctx, gradOutputTr, gradBiasCasted));
        DIOPI_CALL(dataTypeCast(ctx, biasGradTensor, gradBiasCasted))
    }

    DIOPI_CALL(transpose(ctx, gradInputTr, gradInputCasted, {0, 3, 1, 2}));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputCasted));
    DIOPI_CALL(transpose(ctx, gradWeightTr, gradWeightCasted, {0, 3, 1, 2}));
    DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, gradWeightCasted));
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

    DiopiTensor inputTr, weightTr, outputTr;
    DIOPI_CALL(transpose(ctx, inputCasted, inputTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, weightCasted, weightTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, outputCasted, outputTr, {0, 2, 3, 1}));

    DIOPI_CALL(convBackwardData(ctx, inputTr, outputTr, weightTr, stride, padding, dilation, groups));

    if (biasTensor.defined()) {
        cnnlHandle_t handle = cnnlHandlePool.get(ctx);
        DiopiTensor biasCasted = biasTensor;
        std::vector<DiopiTensor *> tensors{&biasCasted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        CnnlTensorDesc biasDesc(biasCasted, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc outputDesc(outputTr, CNNL_LAYOUT_ARRAY);
        size_t workspaceSizeBias;
        DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, biasDesc.get(), outputDesc.get(), &workspaceSizeBias));

        void *workspaceBias = nullptr;
        if (workspaceSizeBias != 0) {
            workspaceBias = requiresBuffer(ctx, workspaceSizeBias).data();
        }
        float alpha = 1.0;
        float beta = 1.0;
        DIOPI_CALLCNNL(
            cnnlBiasAdd(handle, &alpha, biasDesc.get(), biasCasted.data(), workspaceBias, workspaceSizeBias, &beta, outputDesc.get(), outputTr.data()));
    }

    DIOPI_CALL(transpose(ctx, outputTr, outputCasted, {0, 3, 1, 2}));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outputCasted));
    return diopiSuccess;
}

extern "C" diopiError_t diopiConvTranspose2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                                     diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                                     diopiConstTensorHandle_t weight, diopiSize_t *biasSizes, diopiSize_t stride, diopiSize_t padding,
                                                     diopiSize_t dilation, diopiSize_t outputPadding, int64_t groups) {
    DiopiTensor gradInputTensor = DiopiTensor(gradInput);
    DiopiTensor gradWeightTensor = DiopiTensor(gradWeight);
    DiopiTensor gradOutputTensor = DiopiTensor(gradOutput);
    DiopiTensor inputTensor = DiopiTensor(input);
    DiopiTensor weightTensor = DiopiTensor(weight);
    DiopiTensor gradBiasTensor = DiopiTensor(gradBias);

    DiopiTensor gradInputCasted = gradInputTensor;
    DiopiTensor gradWeightCasted = gradWeightTensor;
    DiopiTensor gradOutputCasted = gradOutputTensor;
    DiopiTensor inputCasted = inputTensor;
    DiopiTensor weightCasted = weightTensor;

    std::vector<DiopiTensor *> tensors{&gradInputCasted, &gradWeightCasted, &gradOutputCasted, &inputCasted, &weightCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    DiopiTensor gradInputTr, gradWeightTr, gradOutputTr, inputTr, weightTr;
    DIOPI_CALL(transpose(ctx, gradInputCasted, gradInputTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, gradWeightCasted, gradWeightTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, gradOutputCasted, gradOutputTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, inputCasted, inputTr, {0, 2, 3, 1}));
    DIOPI_CALL(transpose(ctx, weightCasted, weightTr, {0, 2, 3, 1}));

    DIOPI_CALL(convForward(ctx, gradOutputTr, weightTr, {}, gradInputTr, stride, padding, dilation, groups));
    DIOPI_CALL(convBackwardFilter(ctx, inputTr, gradWeightTr, gradOutputTr, stride, padding, dilation, groups));
    if (gradBiasTensor.defined()) {
        DiopiTensor gradBiasCasted = gradBiasTensor;
        std::vector<DiopiTensor *> tensors{&gradBiasCasted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        DIOPI_CALL(convBackwardBias(ctx, gradOutputTr, gradBiasCasted));
        DIOPI_CALL(dataTypeCast(ctx, gradBiasTensor, gradBiasCasted))
    }

    DIOPI_CALL(transpose(ctx, gradInputTr, gradInputCasted, {0, 3, 1, 2}));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputCasted));
    DIOPI_CALL(transpose(ctx, gradWeightTr, gradWeightCasted, {0, 3, 1, 2}));
    DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, gradWeightCasted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl