/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    DiopiTensor targetTensor(target);
    DiopiTensor weightTensor(weight);
    if (!weightTensor.defined()) {
        weightTensor = ones(ctx, {inputTensor.shape()[1]}, inputTensor.dtype());
    }
    DIOPI_CHECK(inputTensor.numel() != 0, "input tensor is empty")
    DIOPI_CHECK(inputTensor.isContiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(weightTensor.isContiguous(), "weight tensor should be contiguous");
    DIOPI_CHECK(targetTensor.isContiguous(), "input tensor should be contiguous");
    if (ReductionMean == reduction || ReductionSum == reduction) {
        DIOPI_CHECK(outputTensor.dim() <= 1, "output.dim should be <= 1 when the redcution is %s.", reductionStr(reduction));
    }

    std::vector<DiopiTensor*> pTensors{&inputTensor, &weightTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outputTmpTensor = outputTensor;
    if (inputTensor.dtype() != outputTensor.dtype()) {
        outputTmpTensor = requiresTensor(ctx, outputTensor.shape(), inputTensor.dtype());
    }

    if (targetTensor.dtype() != diopi_dtype_int32) {
        DIOPI_CALL(dataTypeCast(ctx, targetTensor, diopi_dtype_int32));
    }

    auto inputContiguous = inputTensor;

    auto dim = inputTensor.dim();
    if (dim == 2 || dim == 1) {
        DIOPI_CHECK(targetTensor.dim() == 1, "1D target_tr tensor expected, multi-target_tr not supported");
        DIOPI_CHECK(inputTensor.shape()[0] == targetTensor.shape()[0], "size mismatch ");
        DIOPI_CHECK(!weightTensor.defined() || weightTensor.numel() == inputTensor.shape()[1],
                    "weight_tr tensor should be defined either for all classes or no classes");
    } else if (dim == 4) {
        inputContiguous = inputTensor.contiguous(ctx, MemoryFormat::ChannelsLast);
        DIOPI_CALL(cnnlTranspose(ctx, handle, inputTensor, inputContiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));
    } else if (dim == 3) {
        int64_t inputLastSize = 1;
        for (int i = 2; i < inputTensor.dim(); ++i) {
            inputLastSize *= inputTensor.shape()[i];
        }
        inputTensor.view({inputTensor.shape()[0], inputTensor.shape()[1], 1, inputLastSize});

        inputContiguous = inputTensor.contiguous(ctx, MemoryFormat::ChannelsLast);
        DIOPI_CALL(cnnlTranspose(ctx, handle, inputTensor, inputContiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));
    } else {
        DIOPI_CHECK(false, "unexpected input tensor dim")
    }

    auto inputSize = inputContiguous.shape();
    int c = inputSize[1];
    int n = std::accumulate(inputSize.begin(), inputSize.end(), 1, std::multiplies<>()) / c;
    DIOPI_CHECK(n == targetTensor.numel(), "Target size need be equal as input N*H*W.");
    DIOPI_CHECK(c == weightTensor.numel(), "Weight size need be equal as input C.");
    std::vector<int64_t> outputSize(inputSize.begin(), inputSize.end());

    cnnlNlllossAlgorithm_t reductionMode;
    switch (reduction) {
        case 0: {
            reductionMode = CNNL_REDUCTION_NONE;
            outputSize = {n};
            break;
        }
        case 1: {
            reductionMode = CNNL_REDUCTION_MEAN;
            outputSize = {1};
            break;
        }
        case 2: {
            reductionMode = CNNL_REDUCTION_SUM;
            outputSize = {1};
            break;
        }
        default:
            DIOPI_CHECK(false, "unexpected nll_loss reduciton mode");
    }
    auto totalWeightTensor = requiresTensor(ctx, {1}, weightTensor.dtype());
    diopiScalar_t scalar({weightTensor.dtype(), static_cast<double>(targetTensor.numel())});
    DIOPI_CALL(diopiFill(ctx, totalWeightTensor.tensorHandle(), &scalar));

    outputTmpTensor.asStrided(outputSize, {1});
    CnnlTensorDesc inputDesc;
    CnnlTensorDesc targetDesc;
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc twDesc(totalWeightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTmpTensor, CNNL_LAYOUT_ARRAY);
    inputDesc.set(inputContiguous, CNNL_LAYOUT_ARRAY, {n, c});
    targetDesc.set(targetTensor, CNNL_LAYOUT_ARRAY, {n});

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetNlllossWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlNlllossForward(handle,
                                      reductionMode,
                                      workspacePtr,
                                      workspaceSize,
                                      inputDesc.get(),
                                      inputContiguous.data(),
                                      targetDesc.get(),
                                      targetTensor.data(),
                                      static_cast<int>(ignoreIndex),
                                      weightDesc.get(),
                                      weightTensor.data(),
                                      twDesc.get(),
                                      totalWeightTensor.data(),
                                      outputDesc.get(),
                                      outputTmpTensor.data()));

    if (outputTmpTensor.dtype() != outputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outputTensor, outputTmpTensor));
    }

    return diopiSuccess;
}

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor targetTensor(target);
    DiopiTensor weightTensor(weight);

    if (!weightTensor.defined()) {
        weightTensor = ones(ctx, {inputTensor.shape()[1]}, inputTensor.dtype());
    }

    DIOPI_CHECK(inputTensor.numel() != 0, "input tensor is empty")
    DIOPI_CHECK(inputTensor.isContiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(weightTensor.isContiguous(), "weight tensor should be contiguous");
    DIOPI_CHECK(targetTensor.isContiguous(), "input tensor should be contiguous");
    if (ReductionMean == reduction || ReductionSum == reduction) {
        DIOPI_CHECK(gradOutputTensor.dim() <= 1, "grad_output.dim should be <= 1 when the redcution is %s.", reductionStr(reduction));
    }
    std::vector<DiopiTensor*> pTensors{&gradOutputTensor, &weightTensor, &inputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    if (targetTensor.dtype() != diopi_dtype_int32) {
        DIOPI_CALL(dataTypeCast(ctx, targetTensor, diopi_dtype_int32));
    }

    auto inputContiguous = inputTensor;

    auto dim = inputTensor.dim();
    if (dim == 2 || dim == 1) {
        DIOPI_CHECK(targetTensor.dim() == 1, "1D target_tr tensor expected, multi-target_tr not supported");
        DIOPI_CHECK(inputTensor.shape()[0] == targetTensor.shape()[0], "size mismatch ");
        DIOPI_CHECK(!weightTensor.defined() || weightTensor.numel() == inputTensor.shape()[1],
                    "weight_tr tensor should be defined either for all classes or no classes");
    } else if (dim == 4) {
        inputContiguous = inputTensor.contiguous(ctx, MemoryFormat::ChannelsLast);
        DIOPI_CALL(cnnlTranspose(ctx, handle, inputTensor, inputContiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));
    } else if (dim == 3) {
        int64_t inputLastSize = 1;
        for (int i = 2; i < inputTensor.dim(); ++i) {
            inputLastSize *= inputTensor.shape()[i];
        }
        inputTensor.view({inputTensor.shape()[0], inputTensor.shape()[1], 1, inputLastSize});

        inputContiguous = inputTensor.contiguous(ctx, MemoryFormat::ChannelsLast);
        DIOPI_CALL(cnnlTranspose(ctx, handle, inputTensor, inputContiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));
    } else {
        DIOPI_CHECK(false, "unexpected input tensor dim")
    }

    auto inputSize = inputContiguous.shape();
    int c = inputSize[1];
    int n = std::accumulate(inputSize.begin(), inputSize.end(), 1, std::multiplies<>()) / c;
    DIOPI_CHECK(n == targetTensor.numel(), "Target size need be equal as input N*H*W.");
    DIOPI_CHECK(c == weightTensor.numel(), "Weight size need be equal as input C.");

    cnnlNlllossAlgorithm_t reductionMode;
    switch (reduction) {
        case 0:
            reductionMode = CNNL_REDUCTION_NONE;
            break;
        case 1:
            reductionMode = CNNL_REDUCTION_MEAN;
            break;
        case 2:
            reductionMode = CNNL_REDUCTION_SUM;
            break;
        default:
            DIOPI_CHECK(false, "unexpected nll_loss reduciton mode");
    }

    auto gradInputRealTensor = requiresTensor(ctx, {n, c}, inputContiguous.dtype());

    auto totalWeightTensor = requiresTensor(ctx, {1}, weightTensor.dtype());
    diopiScalar_t scalar({weightTensor.dtype(), static_cast<double>(targetTensor.numel())});
    DIOPI_CALL(diopiFill(ctx, totalWeightTensor.tensorHandle(), &scalar));

    CnnlTensorDesc gradOutputDesc;
    CnnlTensorDesc targetDesc;
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc twDesc(totalWeightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputRealTensor, CNNL_LAYOUT_ARRAY);
    targetDesc.set(targetTensor, CNNL_LAYOUT_ARRAY, {n});
    reduction == 0 ? gradOutputDesc.set(gradOutputTensor, CNNL_LAYOUT_ARRAY, {n}) : gradOutputDesc.set(gradOutputTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlNlllossBackward(handle,
                                       reductionMode,
                                       gradOutputDesc.get(),
                                       gradOutputTensor.data(),
                                       targetDesc.get(),
                                       targetTensor.data(),
                                       static_cast<int>(ignoreIndex),
                                       weightDesc.get(),
                                       weightTensor.data(),
                                       twDesc.get(),
                                       totalWeightTensor.data(),
                                       gradInputDesc.get(),
                                       gradInputRealTensor.data()));
    if (dim > 2) {
        // NHWC -> NCHW and dealing with data type
        gradInputRealTensor.view(inputContiguous.shape());
        gradInputTensor.view(inputContiguous.shape());

        DiopiTensor gradInputTmpTensor = gradInputTensor;
        if (gradInputTensor.dtype() != gradInputRealTensor.dtype()) {
            gradInputTmpTensor = requiresTensor(ctx, gradInputTensor.shape(), gradInputRealTensor.dtype());
        }

        DIOPI_CALL(cnnlTranspose(ctx, handle, gradInputRealTensor, gradInputTmpTensor, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW));

        if (gradInputTmpTensor.dtype() != gradInputTensor.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTmpTensor));
        }

    } else {
        DIOPI_CALL(diopiCopyInp(ctx, gradInputRealTensor.tensorHandle(), gradInputTensor.tensorHandle()));
    }

    return diopiSuccess;
}

diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                   diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    DiopiTensor inputTensor(input);
    DiopiTensor targetTensor(target);

    DIOPI_CHECK(labelSmoothing == 0, "Param label_smoothing is not supported by cnnl")
    DIOPI_CHECK(targetTensor.dim() == inputTensor.dim() - 1, "Probabilities for each class are not supported by cnnl");

    auto logTensor = requiresTensor(ctx, inputTensor.shape(), inputTensor.dtype());
    DIOPI_CALL(diopiLogSoftmax(ctx, logTensor.tensorHandle(), input, 1));
    DIOPI_CALL(diopiNLLLoss(ctx, out, logTensor.tensorHandle(), target, weight, reduction, ignoreIndex));
    return diopiSuccess;
}
diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                           diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    DiopiTensor inputTensor(input);
    DiopiTensor targetTensor(target);
    DiopiTensor gradInputTensor(gradInput);

    DIOPI_CHECK(labelSmoothing == 0, "param label_smoothing is not supported")
    DIOPI_CHECK(targetTensor.dim() == inputTensor.dim() - 1, "Probabilities for each class are not supported");

    auto logTensor = requiresTensor(ctx, inputTensor.shape(), inputTensor.dtype());
    auto gradTmpTensor = requiresTensor(ctx, gradInputTensor.shape(), gradInputTensor.dtype());

    DIOPI_CALL(diopiLogSoftmax(ctx, logTensor.tensorHandle(), input, 1));
    // for nll loss backward, `input` should be logsoftmax out.
    DIOPI_CALL(diopiNLLLossBackward(ctx, gradTmpTensor.tensorHandle(), gradOutput, logTensor.tensorHandle(), target, weight, reduction, ignoreIndex));
    // for softmax backward, `output` should be logsoftmax out
    DIOPI_CALL(diopiLogSoftmaxBackward(ctx, gradInput, gradTmpTensor.tensorHandle(), logTensor.tensorHandle(), 1));
    return diopiSuccess;
}

diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiReduction_t reduction) {
    DiopiTensor trInput(input);
    DiopiTensor trTarget(target);
    DiopiTensor trOut(out);
    std::vector<DiopiTensor*> pTensors{&trInput, &trTarget};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    cnnlMSELossReduction_t cnnlReduction;
    if (reduction == ReductionMean) {
        cnnlReduction = CNNL_MSE_LOSS_MEAN;
        DIOPI_CHECK(trOut.dim() == 0, "Output dim must be 0.");
    } else if (reduction == ReductionSum) {
        cnnlReduction = CNNL_MSE_LOSS_SUM;
        DIOPI_CHECK(trOut.dim() == 0, "Output dim must be 0.");
    } else {
        cnnlReduction = CNNL_MSE_LOSS_NONE;
        DIOPI_CHECK(trOut.dim() == trInput.dim(), "Output dim must be the same as input.");
    }

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descTarget(trTarget, layout);
    CnnlTensorDesc descOut;
    DiopiTensor trOutTmp;
    if (trInput.dtype() == trOut.dtype()) {
        trOutTmp = trOut;
        descOut.set(trOut, layout);
    } else {
        trOutTmp = requiresTensor(ctx, vec2diopiSizeT(trOut.shape()), trInput.dtype());
        descOut.set(trOutTmp, CNNL_LAYOUT_ARRAY);
    }

    DIOPI_CALLCNNL(cnnlMSELoss(handle, cnnlReduction, descInput.get(), trInput.data(), descTarget.get(), trTarget.data(), descOut.get(), trOutTmp.data()));
    if (trOutTmp.dtype() != trOut.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, trOut, trOutTmp));
    }
    return diopiSuccess;
}

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    DiopiTensor trInput(input);
    DiopiTensor trGradOutput(gradOutput);
    DiopiTensor trTarget(target);
    DiopiTensor trGradInput(gradInput);

    std::vector<DiopiTensor*> pTensors{&trInput, &trGradOutput, &trTarget};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    cnnlMSELossReduction_t cnnlReduction;
    if (reduction == ReductionMean) {
        cnnlReduction = CNNL_MSE_LOSS_MEAN;
        DIOPI_CHECK(trGradOutput.dim() == 0, "Grad output dim must be 0.");
    } else if (reduction == ReductionSum) {
        cnnlReduction = CNNL_MSE_LOSS_SUM;
        DIOPI_CHECK(trGradOutput.dim() == 0, "Grad output dim must be 0.");
    } else {
        cnnlReduction = CNNL_MSE_LOSS_NONE;
        DIOPI_CHECK(trGradOutput.dim() == trInput.dim(), "Output dim must be the same as input.");
    }

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descTarget(trTarget, layout);
    CnnlTensorDesc descGradOutput(trGradOutput, layout);
    CnnlTensorDesc descGradInput;
    DiopiTensor trGradInputTmp;
    CnnlTensorDesc descGradInputTmp;
    if (trInput.dtype() == trGradInput.dtype()) {
        trGradInputTmp = trGradInput;
        descGradInput.set(trGradInput, layout);
    } else {
        trGradInputTmp = requiresTensor(ctx, vec2diopiSizeT(trGradInput.shape()), trInput.dtype());
        descGradInput.set(trGradInputTmp, CNNL_LAYOUT_ARRAY);
    }

    DIOPI_CALLCNNL(cnnlMSELossBackward(handle,
                                       cnnlReduction,
                                       descInput.get(),
                                       trInput.data(),
                                       descTarget.get(),
                                       trTarget.data(),
                                       descGradOutput.get(),
                                       trGradOutput.data(),
                                       descGradInput.get(),
                                       trGradInputTmp.data()));
    if (trGradInputTmp.dtype() != trGradInput.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, trGradInput, trGradInputTmp));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
