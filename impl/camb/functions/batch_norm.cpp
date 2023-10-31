/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/debug.hpp"
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
    DiopiTensor outTr(out);

    DIOPI_CHECK(inputTr.shape().size() >= 2, "input's dim should be greater than 2.")

    auto dim = inputTr.dim();
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == outTr.dim(), "Input dim != out dim");

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

    DIOPI_CHECK(saveMeanTr.defined(), "saveMean is not defined.");
    DIOPI_CHECK(saveInvstdTr.defined(), "saveInvstd is not defined.");

    cnnlTensorLayout_t layout;

    if (5 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "the dim of input is 5, but the memory format of it is not channelast3d.");
        DIOPI_CHECK(outTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "the dim of out is 5, but the memory format of it is not channelast3d.");
        layout = CNNL_LAYOUT_NDHWC;
    } else if (4 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "the dim of input is 4, but the memory format of it is not channelast.");
        DIOPI_CHECK(outTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "the dim of out is 4, but the memory format of it is not channelast.");
        layout = CNNL_LAYOUT_NHWC;
    } else if (3 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "the dim of input is 3, but the memory format of it is not channelast1d.");
        DIOPI_CHECK(outTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "the dim of out is 3, but the memory format of it is not channelast1d.");
        layout = CNNL_LAYOUT_NLC;
    } else if (2 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::Contiguous), "the dim of input is 2, but the memory format of it is not contiguous.");
        DIOPI_CHECK(outTr.isContiguous(diopiMemoryFormat_t::Contiguous), "the dim of out is 2, but the memory format of it is not contiguous.");
        layout = CNNL_LAYOUT_NC;
    }

    std::vector<DiopiTensor*> pTensors{&inputTr, &weightTr, &biasTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor runningMeanTmpTr = runningMeanTr;
    if (runningMeanTr.defined() && runningMeanTr.dtype() != inputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, runningMeanTmpTr, inputTr.dtype()));
    }

    DiopiTensor runningVarTmpTr = runningVarTr;
    if (runningVarTr.defined() && runningVarTr.dtype() != inputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, runningVarTmpTr, inputTr.dtype()));
    }
    DiopiTensor saveInvstdTmpTr = saveInvstdTr;
    if (saveInvstdTr.dtype() != inputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, saveInvstdTmpTr, inputTr.dtype()));
    }
    DiopiTensor saveMeanTmpTr = saveMeanTr;
    if (saveMeanTr.dtype() != inputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, saveMeanTmpTr, inputTr.dtype()));
    }

    DiopiTensor outTmpTr = outTr;
    if (outTr.dtype() != inputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTr, inputTr.dtype()));
    }

    CnnlTensorDesc weightBiasMeanVarDesc(weightTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc outDesc(outTr, layout);

    if (training) {
        size_t workspaceSize = 0;
        DIOPI_CALLCNNL(cnnlGetBatchNormForwardWorkspaceSize(handle, inputDesc.get(), &workspaceSize));

        void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

        // set activition part to default
        cnnlActivationMode_t activeMode = CNNL_ACTIVATION_IDENTITY;
        cnnlActivationDescriptor_t activationDesc = nullptr;
        DIOPI_CALLCNNL(cnnlCreateActivationDescriptor(&activationDesc));
        cnnlSetActivationDescriptor_v5(activationDesc, activeMode, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);
        DIOPI_CALLCNNL(cnnlBatchNormForwardTraining_v2(handle,
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
                                                       runningMeanTmpTr.defined() ? runningMeanTr.data() : nullptr,
                                                       runningVarTmpTr.defined() ? runningVarTr.data() : nullptr,
                                                       static_cast<float>(eps),
                                                       static_cast<float>(momentum),
                                                       outDesc.get(),
                                                       outTmpTr.data(),
                                                       saveMeanTmpTr.data(),
                                                       saveInvstdTmpTr.data(),
                                                       workspacePtr,
                                                       workspaceSize,
                                                       nullptr,
                                                       0));
    } else {
        DIOPI_CALLCNNL(cnnlBatchNormForwardInference(handle,
                                                     nullptr,
                                                     nullptr,
                                                     inputDesc.get(),
                                                     inputTr.data(),
                                                     weightBiasMeanVarDesc.get(),
                                                     weightTr.data(),
                                                     biasTr.data(),
                                                     runningMeanTmpTr.defined() ? runningMeanTr.data() : nullptr,
                                                     runningVarTmpTr.defined() ? runningVarTr.data() : nullptr,
                                                     static_cast<float>(eps),
                                                     outDesc.get(),
                                                     outTmpTr.data()));
    }
    if (runningMeanTr.defined() && runningMeanTmpTr.dtype() != runningMeanTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, runningMeanTr, runningMeanTmpTr));
    }
    if (runningVarTr.defined() && runningVarTmpTr.dtype() != runningVarTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, runningVarTr, runningVarTmpTr));
    }
    if (saveInvstdTmpTr.dtype() != saveInvstdTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, saveInvstdTr, saveInvstdTmpTr));
    }

    if (saveMeanTmpTr.dtype() != saveMeanTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, saveMeanTr, saveMeanTmpTr));
    }
    if (outTmpTr.dtype() != outTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTr, outTmpTr));
    }

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

    cnnlTensorLayout_t layout;

    if (5 == dim) {
        DIOPI_CHECK(gradOutputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d),
                    "the dim of gradOutput is 5, but the memory format of it is not channelast3d.");
        DIOPI_CHECK(gradInputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d),
                    "the dim of gradInput is 5, but the memory format of it is not channelast3d.");
        layout = CNNL_LAYOUT_NDHWC;
    } else if (4 == dim) {
        DIOPI_CHECK(gradOutputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "the dim of gradOutput is 4, but the memory format of it is not channelast.");
        DIOPI_CHECK(gradInputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "the dim of gradInput is 4, but the memory format of it is not channelast.");
        layout = CNNL_LAYOUT_NHWC;
    } else if (3 == dim) {
        DIOPI_CHECK(gradOutputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d),
                    "the dim of gradOutput is 3, but the memory format of it is not channelast1d.");
        DIOPI_CHECK(gradInputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d),
                    "the dim of gradInput is 3, but the memory format of it is not channelast1d.");
        layout = CNNL_LAYOUT_NLC;
    } else if (2 == dim) {
        DIOPI_CHECK(gradOutputTr.isContiguous(diopiMemoryFormat_t::Contiguous), "the dim of gradOutput is 2, but the memory format of it is not contiguous.");
        DIOPI_CHECK(gradInputTr.isContiguous(diopiMemoryFormat_t::Contiguous), "the dim of gradInput is 2, but the memory format of it is not contiguous.");
        layout = CNNL_LAYOUT_NC;
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
    DiopiTensor gradInputTmpTr = gradInputTr;
    if (gradInputTr.dtype() != gradOutputTr.dtype()) {
        gradInputTmpTr = requiresTensor(ctx, gradInputTr.shape(), gradOutputTr.dtype());
    }
    /* Transpose */
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc gradOutputDesc(gradOutputTr, layout);
    CnnlTensorDesc gradInputDesc(gradInputTr, layout);
    CnnlTensorDesc weightBiasMeanVarDesc(weightTr, CNNL_LAYOUT_ARRAY);

    // set activition part
    cnnlBatchNormMode_t mode = CNNL_BATCHNORM_SPATIAL;
    cnnlBatchNormOps_t bnOps = CNNL_BATCHNORM_OPS_BN;
    cnnlActivationMode_t activeMode = CNNL_ACTIVATION_IDENTITY;

    cnnlActivationDescriptor_t activationDesc = nullptr;
    DIOPI_CALLCNNL(cnnlCreateActivationDescriptor(&activationDesc));
    cnnlSetActivationDescriptor_v5(activationDesc, activeMode, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);

    if (training) {
        // get workspace
        size_t workspaceSize = 0;
        DIOPI_CALLCNNL(cnnlGetBatchNormBackwardWorkspaceSize(handle, inputDesc.get(), &workspaceSize));

        void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

        DIOPI_CALLCNNL(cnnlBatchNormBackward_v2(handle,
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
                                                gradInputTr.data(),
                                                gradWeightTmpTr.data(),
                                                gradBiasTmpTr.data(),
                                                workspacePtr,
                                                workspaceSize,
                                                nullptr,
                                                0));
    } else {
        size_t workspaceSize = 0;
        DIOPI_CALLCNNL(cnnlGetFrozenBatchNormBackwardWorkspaceSize(handle, inputDesc.get(), &workspaceSize));

        void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

        DIOPI_CALLCNNL(cnnlFrozenBatchNormBackward_v2(handle,
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
                                                      gradInputTr.data(),
                                                      gradWeightTmpTr.data(),
                                                      gradBiasTmpTr.data()));
    }
    if (gradInputTmpTr.dtype() != gradInputTr.dtype()) {
        dataTypeCast(ctx, gradInputTr, gradInputTmpTr);
    }
    if (gradWeightTmpTr.dtype() != gradWeightTr.dtype()) {
        dataTypeCast(ctx, gradWeightTr, gradWeightTmpTr);
    }
    if (gradBiasTmpTr.dtype() != gradBiasTr.dtype()) {
        dataTypeCast(ctx, gradBiasTr, gradBiasTmpTr);
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
