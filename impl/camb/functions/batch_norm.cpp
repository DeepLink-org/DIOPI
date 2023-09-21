/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t runningMean,
                            diopiTensorHandle_t runningVar, bool training, double momentum, double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor saveMeanTr(saveMean);
    DiopiTensor saveInvstdTr(saveInvstd);
    DiopiTensor inputTr(input);
    DiopiTensor weightTr(weight);
    DiopiTensor biasTr(bias);
    DiopiTensor runningMeanTr(runningMean);
    DiopiTensor runningVarTr(runningVar);
    DiopiTensor outputTr(out);

    DiopiTensor runningMeanTrOrigin(runningMean);
    DiopiTensor runningVarTrOrigin(runningVar);

    DIOPI_CHECK(inputTr.shape().size() >= 2, "input's dim should be greater than 2.")
    /* Some basic check */
    if (runningMeanTr.defined() && runningVarTr.defined()) {
        DIOPI_CHECK(runningMeanTr.dtype() == runningVarTr.dtype(), "running_mean and running_var need to have the same data types");
    }
    auto dim = inputTr.dim();
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == outputTr.dim(), "Input dim != out dim");

    if (!weightTr.defined()) {
        diopiScalar_t val = constructDiopiScalarT(diopi_dtype_float32, 1.0f);
        weightTr = requiresTensor(ctx, {inputTr.shape()[1]}, inputTr.dtype());
        DIOPI_CALL(diopiFill(ctx, weightTr.tensorHandle(), &val))
    }
    if (!biasTr.defined()) {
        diopiScalar_t val = constructDiopiScalarT(diopi_dtype_float32, 0.0f);
        biasTr = requiresTensor(ctx, {inputTr.shape()[1]}, inputTr.dtype());
        DIOPI_CALL(diopiFill(ctx, biasTr.tensorHandle(), &val))
    }

    if (3 == dim) {
        inputTr.unsqueeze(3);
        outputTr.view(inputTr.shape());
    }
    if (2 == dim) {
        inputTr.unsqueeze(2);
        inputTr.unsqueeze(3);
        outputTr.view(inputTr.shape());
    }

    std::vector<DiopiTensor*> pTensors{&inputTr, &weightTr, &biasTr};
    if (runningMeanTr.defined()) {
        pTensors.push_back(&runningMeanTr);
    }
    if (runningVarTr.defined()) {
        pTensors.push_back(&runningVarTr);
    }
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // Note: 1. output.dtype = input.dtype  2. channelsLast format
    diopiMemoryFormat_t memoryFormat = inputTr.dim() == 4 ? diopiMemoryFormat_t::ChannelsLast : diopiMemoryFormat_t::ChannelsLast3d;
    DiopiTensor outputTmpTr = requiresTensor(ctx, outputTr.shape(), inputTr.dtype(), memoryFormat);

    /* Transpose to channels last */
    DIOPI_CALL(contiguous(ctx, inputTr, memoryFormat));

    CnnlTensorDesc weightBiasMeanVarDesc(weightTr, CNNL_LAYOUT_ARRAY);
    cnnlTensorLayout_t layout = inputTr.dim() == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc outputDesc(outputTmpTr, layout);

    if (training) {
        size_t workspaceSize = 0;
        DIOPI_CALL_CNNL(cnnlGetBatchNormForwardWorkspaceSize(handle, inputDesc.get(), &workspaceSize));

        void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

        // set activition part to default
        cnnlActivationMode_t activeMode = CNNL_ACTIVATION_IDENTITY;
        cnnlActivationDescriptor_t activationDesc = nullptr;
        DIOPI_CALL_CNNL(cnnlCreateActivationDescriptor(&activationDesc));
        cnnlSetActivationDescriptor_v5(activationDesc, activeMode, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);
        DIOPI_CALL_CNNL(cnnlBatchNormForwardTraining_v2(handle,
                                                        activationDesc,
                                                        CNNL_BATCHNORM_SPATIAL,
                                                        CNNL_BATCHNORM_OPS_BN,
                                                        nullptr,
                                                        nullptr,
                                                        inputDesc.get(),
                                                        inputTr.data(),
                                                        nullptr,
                                                        nullptr,
                                                        weightBiasMeanVarDesc.get(),
                                                        weightTr.data(),
                                                        biasTr.data(),
                                                        runningMeanTr.defined() ? runningMeanTr.data() : nullptr,
                                                        runningVarTr.defined() ? runningVarTr.data() : nullptr,
                                                        static_cast<float>(eps),
                                                        static_cast<float>(momentum),
                                                        outputDesc.get(),
                                                        outputTmpTr.data(),
                                                        saveMeanTr.data(),
                                                        saveInvstdTr.data(),
                                                        workspacePtr,
                                                        workspaceSize,
                                                        nullptr,
                                                        0));
    } else {
        DIOPI_CALL_CNNL(cnnlBatchNormForwardInference(handle,
                                                      nullptr,
                                                      nullptr,
                                                      inputDesc.get(),
                                                      inputTr.data(),
                                                      weightBiasMeanVarDesc.get(),
                                                      weightTr.data(),
                                                      biasTr.data(),
                                                      runningMeanTr.defined() ? runningMeanTr.data() : nullptr,
                                                      runningVarTr.defined() ? runningVarTr.data() : nullptr,
                                                      static_cast<float>(eps),
                                                      outputDesc.get(),
                                                      outputTmpTr.data()));
    }

    // channels last -> contiguous
    DIOPI_CALL(contiguous(ctx, outputTmpTr, diopiMemoryFormat_t::Contiguous));
    // Copy back to origin
    DIOPI_CALL(diopiCopyInp(ctx, outputTmpTr.tensorHandle(), outputTr.tensorHandle()));
    DIOPI_CALL(diopiCopyInp(ctx, runningMeanTr.tensorHandle(), runningMeanTrOrigin.tensorHandle()));
    DIOPI_CALL(diopiCopyInp(ctx, runningVarTr.tensorHandle(), runningVarTrOrigin.tensorHandle()));

    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t runningMean, diopiConstTensorHandle_t runningVar, diopiConstTensorHandle_t saveMean,
                                    diopiConstTensorHandle_t saveInvstd, bool training, double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor gradInputTr(gradInput);
    DiopiTensor gradWeightTr(gradWeight);
    DiopiTensor gradBiasTr(gradBias);
    DiopiTensor inputTr(input);
    DiopiTensor weightTr(weight);
    DiopiTensor runningMeanTr(runningMean);
    DiopiTensor runningVarTr(runningVar);
    DiopiTensor saveMeanTr(saveMean);
    DiopiTensor saveInvstdTr(saveInvstd);

    DiopiTensor gradOutputTr(gradOutput);

    if (runningMeanTr.defined() && runningVarTr.defined()) {
        DIOPI_CHECK(runningMeanTr.dtype() == runningVarTr.dtype(), "running_mean and running_var need to have the same data types");
    }
    auto dim = inputTr.dim();
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");

    if (3 == dim) {
        inputTr.unsqueeze(3);
        gradOutputTr.unsqueeze(3);
        gradInputTr.view(inputTr.shape());
    }
    if (2 == dim) {
        inputTr.unsqueeze(2);
        inputTr.unsqueeze(3);
        gradOutputTr.unsqueeze(2);
        gradOutputTr.unsqueeze(3);
        gradInputTr.view(inputTr.shape());
    }

    std::vector<DiopiTensor*> pTensors{&gradOutputTr, &inputTr, &weightTr};
    if (runningMeanTr.defined()) {
        pTensors.push_back(&runningMeanTr);
    }
    if (runningVarTr.defined()) {
        pTensors.push_back(&runningVarTr);
    }
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor gradWeightTmpTr = gradWeightTr;
    if (gradWeightTr.dtype() != gradOutputTr.dtype()) {
        gradWeightTmpTr = requiresTensor(ctx, gradWeightTr.shape(), gradOutputTr.dtype());
    }
    DiopiTensor gradBiasTmpTr = gradBiasTr;
    if (gradBiasTr.dtype() != gradOutputTr.dtype()) {
        gradBiasTmpTr = requiresTensor(ctx, gradBiasTr.shape(), gradOutputTr.dtype());
    }

    /* Transpose */
    diopiMemoryFormat_t memoryFormat = inputTr.dim() == 4 ? diopiMemoryFormat_t::ChannelsLast : diopiMemoryFormat_t::ChannelsLast3d;
    DIOPI_CALL(contiguous(ctx, inputTr, memoryFormat));
    DIOPI_CALL(contiguous(ctx, gradOutputTr, memoryFormat));

    // Note: 1. output.dtype = input.dtype  2. channelsLast format
    DiopiTensor gradInputTmpTr = requiresTensor(ctx, gradInputTr.shape(), gradOutputTr.dtype(), memoryFormat);

    cnnlTensorLayout_t layout = inputTr.dim() == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc gradOutputDesc(gradOutputTr, layout);
    CnnlTensorDesc gradInputDesc(gradInputTmpTr, layout);
    CnnlTensorDesc weightBiasMeanVarDesc(weightTr, CNNL_LAYOUT_ARRAY);

    // set activition part
    cnnlBatchNormMode_t mode = CNNL_BATCHNORM_SPATIAL;
    cnnlBatchNormOps_t bnOps = CNNL_BATCHNORM_OPS_BN;
    cnnlActivationMode_t activeMode = CNNL_ACTIVATION_IDENTITY;

    cnnlActivationDescriptor_t activationDesc = nullptr;
    DIOPI_CALL_CNNL(cnnlCreateActivationDescriptor(&activationDesc));
    cnnlSetActivationDescriptor_v5(activationDesc, activeMode, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);

    if (training) {
        // get workspace
        size_t workspaceSize = 0;
        DIOPI_CALL_CNNL(cnnlGetBatchNormBackwardWorkspaceSize(handle, inputDesc.get(), &workspaceSize));

        void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

        DIOPI_CALL_CNNL(cnnlBatchNormBackward_v2(handle,
                                                 activationDesc,
                                                 mode,
                                                 bnOps,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 inputDesc.get(),
                                                 inputTr.data(),
                                                 nullptr,
                                                 nullptr,
                                                 gradOutputDesc.get(),
                                                 gradOutputTr.data(),
                                                 weightBiasMeanVarDesc.get(),
                                                 weightTr.data(),
                                                 nullptr,
                                                 saveMeanTr.defined() ? saveMeanTr.data() : nullptr,
                                                 saveInvstdTr.defined() ? saveInvstdTr.data() : nullptr,
                                                 static_cast<float>(eps),
                                                 nullptr,
                                                 nullptr,
                                                 gradInputDesc.get(),
                                                 gradInputTmpTr.data(),
                                                 gradWeightTmpTr.data(),
                                                 gradBiasTmpTr.data(),
                                                 workspacePtr,
                                                 workspaceSize,
                                                 nullptr,
                                                 0));
    } else {
        size_t workspaceSize = 0;
        DIOPI_CALL_CNNL(cnnlGetFrozenBatchNormBackwardWorkspaceSize(handle, inputDesc.get(), &workspaceSize));

        void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

        DIOPI_CALL_CNNL(cnnlFrozenBatchNormBackward_v2(handle,
                                                       activationDesc,
                                                       mode,
                                                       bnOps,
                                                       inputDesc.get(),
                                                       inputTr.data(),
                                                       nullptr,
                                                       nullptr,
                                                       gradOutputDesc.get(),
                                                       gradOutputTr.data(),
                                                       weightBiasMeanVarDesc.get(),
                                                       weightTr.data(),
                                                       nullptr,
                                                       runningMeanTr.defined() ? runningMeanTr.data() : nullptr,
                                                       runningVarTr.defined() ? runningVarTr.data() : nullptr,
                                                       static_cast<float>(eps),
                                                       workspacePtr,
                                                       workspaceSize,
                                                       nullptr,
                                                       nullptr,
                                                       gradInputDesc.get(),
                                                       gradInputTmpTr.data(),
                                                       gradWeightTmpTr.data(),
                                                       gradBiasTmpTr.data()));
    }

    // Channels last -> contiguous
    DIOPI_CALL(contiguous(ctx, gradInputTmpTr, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(diopiCopyInp(ctx, gradInputTmpTr.tensorHandle(), gradInputTr.tensorHandle()));
    DIOPI_CALL(diopiCopyInp(ctx, gradWeightTmpTr.tensorHandle(), gradWeightTr.tensorHandle()));
    DIOPI_CALL(diopiCopyInp(ctx, gradBiasTmpTr.tensorHandle(), gradBiasTr.tensorHandle()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
