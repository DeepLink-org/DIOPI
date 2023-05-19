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
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(out);
    DiopiTensor targetTr(target);
    DiopiTensor weightTr(weight);
    if (!weightTr.defined()) {
        weightTr = ones(ctx, {inputTr.shape()[1]}, inputTr.dtype());
    }
    DIOPI_CHECK(inputTr.numel() != 0, "input tensor is empty")
    DIOPI_CHECK(inputTr.isContiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(weightTr.isContiguous(), "weight tensor should be contiguous");
    DIOPI_CHECK(targetTr.isContiguous(), "input tensor should be contiguous");
    if (ReductionMean == reduction || ReductionSum == reduction) {
        DIOPI_CHECK(outputTr.dim() <= 1, "output.dim should be <= 1 when the redcution is %s.", reductionStr(reduction));
    }

    std::vector<DiopiTensor*> pTensors{&inputTr, &weightTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outputTmpTr = outputTr;
    if (inputTr.dtype() != outputTr.dtype()) {
        outputTmpTr = requiresTensor(ctx, outputTr.shape(), inputTr.dtype());
    }

    if (targetTr.dtype() != diopi_dtype_int32) {
        DIOPI_CALL(dataTypeCast(ctx, targetTr, diopi_dtype_int32));
    }

    auto inputContiguous = inputTr;

    auto dim = inputTr.dim();
    if (dim == 2 || dim == 1) {
        DIOPI_CHECK(targetTr.dim() == 1, "1D target_tr tensor expected, multi-target_tr not supported");
        DIOPI_CHECK(inputTr.shape()[0] == targetTr.shape()[0], "size mismatch ");
        DIOPI_CHECK(!weightTr.defined() || weightTr.numel() == inputTr.shape()[1],
                    "weight_tr tensor should be defined either for all classes or no classes");
    } else if (dim == 4) {
        inputContiguous = inputTr.contiguous(ctx, MemoryFormat::ChannelsLast);
        DIOPI_CALL(cnnlTranspose(ctx, handle, inputTr, inputContiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));
    } else if (dim == 3) {
        int64_t inputLastSize = 1;
        for (int i = 2; i < inputTr.dim(); ++i) {
            inputLastSize *= inputTr.shape()[i];
        }
        inputTr.reshape({inputTr.shape()[0], inputTr.shape()[1], 1, inputLastSize});

        inputContiguous = inputTr.contiguous(ctx, MemoryFormat::ChannelsLast);
        DIOPI_CALL(cnnlTranspose(ctx, handle, inputTr, inputContiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));
    } else {
        DIOPI_CHECK(false, "unexpected input tensor dim")
    }

    auto inputSize = inputContiguous.shape();
    int c = inputSize[1];
    int n = std::accumulate(inputSize.begin(), inputSize.end(), 1, std::multiplies<int64_t>()) / c;
    DIOPI_CHECK(n == targetTr.numel(), "Target size need be equal as input N*H*W.");
    DIOPI_CHECK(c == weightTr.numel(), "Weight size need be equal as input C.");
    std::vector<int> outputSize(inputSize.begin(), inputSize.end());

    cnnlNlllossAlgorithm_t reductionMode;
    switch (reduction) {
        case 0: {
            reductionMode = CNNL_REDUCTION_NONE;
            outputSize.erase(outputSize.begin() + 1);
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
    auto totalWeightTr = requiresTensor(ctx, {1}, weightTr.dtype());
    diopiScalar_t scalar({weightTr.dtype(), static_cast<double>(targetTr.numel())});
    DIOPI_CALL(diopiFill(ctx, totalWeightTr.tensorHandle(), &scalar));

    CnnlTensorDesc inputDesc;
    CnnlTensorDesc targetDesc;
    CnnlTensorDesc weightDesc(weightTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc twDesc(totalWeightTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc;
    inputDesc.set(inputContiguous, CNNL_LAYOUT_ARRAY, {n, c});
    targetDesc.set(targetTr, CNNL_LAYOUT_ARRAY, {n});
    outputDesc.set(outputTmpTr, CNNL_LAYOUT_ARRAY, outputSize);

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
                                      targetTr.data(),
                                      static_cast<int>(ignoreIndex),
                                      weightDesc.get(),
                                      weightTr.data(),
                                      twDesc.get(),
                                      totalWeightTr.data(),
                                      outputDesc.get(),
                                      outputTmpTr.data()));

    if (outputTmpTr.dtype() != outputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outputTr, outputTmpTr));
    }

    return diopiSuccess;
}

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction,
                                  int64_t ignoreIndex) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTr(input);
    DiopiTensor gradInputTr(gradInput);
    DiopiTensor gradOutputTr(gradOutput);
    DiopiTensor targetTr(target);
    DiopiTensor weightTr(weight);

    if (!weightTr.defined()) {
        weightTr = ones(ctx, {inputTr.shape()[1]}, inputTr.dtype());
    }

    DIOPI_CHECK(inputTr.numel() != 0, "input tensor is empty")
    DIOPI_CHECK(inputTr.isContiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(weightTr.isContiguous(), "weight tensor should be contiguous");
    DIOPI_CHECK(targetTr.isContiguous(), "input tensor should be contiguous");
    if (ReductionMean == reduction || ReductionSum == reduction) {
        DIOPI_CHECK(gradOutputTr.dim() <= 1, "grad_output.dim should be <= 1 when the redcution is %s.", reductionStr(reduction));
    }
    std::vector<DiopiTensor*> pTensors{&gradOutputTr, &weightTr, &inputTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    if (targetTr.dtype() != diopi_dtype_int32) {
        DIOPI_CALL(dataTypeCast(ctx, targetTr, diopi_dtype_int32));
    }

    auto inputContiguous = inputTr;

    auto dim = inputTr.dim();
    if (dim == 2 || dim == 1) {
        DIOPI_CHECK(targetTr.dim() == 1, "1D target_tr tensor expected, multi-target_tr not supported");
        DIOPI_CHECK(inputTr.shape()[0] == targetTr.shape()[0], "size mismatch ");
        DIOPI_CHECK(!weightTr.defined() || weightTr.numel() == inputTr.shape()[1],
                    "weight_tr tensor should be defined either for all classes or no classes");
    } else if (dim == 4) {
        inputContiguous = inputTr.contiguous(ctx, MemoryFormat::ChannelsLast);
        DIOPI_CALL(cnnlTranspose(ctx, handle, inputTr, inputContiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));
    } else if (dim == 3) {
        int64_t inputLastSize = 1;
        for (int i = 2; i < inputTr.dim(); ++i) {
            inputLastSize *= inputTr.shape()[i];
        }
        inputTr.reshape({inputTr.shape()[0], inputTr.shape()[1], 1, inputLastSize});

        inputContiguous = inputTr.contiguous(ctx, MemoryFormat::ChannelsLast);
        DIOPI_CALL(cnnlTranspose(ctx, handle, inputTr, inputContiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));
    } else {
        DIOPI_CHECK(false, "unexpected input tensor dim")
    }

    auto inputSize = inputContiguous.shape();
    int c = inputSize[1];
    int n = std::accumulate(inputSize.begin(), inputSize.end(), 1, std::multiplies<int64_t>()) / c;
    DIOPI_CHECK(n == targetTr.numel(), "Target size need be equal as input N*H*W.");
    DIOPI_CHECK(c == weightTr.numel(), "Weight size need be equal as input C.");

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

    auto gradInputRealTr = requiresTensor(ctx, {n, c}, inputContiguous.dtype());

    auto totalWeightTr = requiresTensor(ctx, {1}, weightTr.dtype());
    diopiScalar_t scalar({weightTr.dtype(), static_cast<double>(targetTr.numel())});
    DIOPI_CALL(diopiFill(ctx, totalWeightTr.tensorHandle(), &scalar));

    CnnlTensorDesc gradOutputDesc(gradOutputTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetDesc;
    CnnlTensorDesc weightDesc(weightTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc twDesc(totalWeightTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputRealTr, CNNL_LAYOUT_ARRAY);
    targetDesc.set(targetTr, CNNL_LAYOUT_ARRAY, {n});

    DIOPI_CALLCNNL(cnnlNlllossBackward(handle,
                                       reductionMode,
                                       gradOutputDesc.get(),
                                       gradOutputTr.data(),
                                       targetDesc.get(),
                                       targetTr.data(),
                                       static_cast<int>(ignoreIndex),
                                       weightDesc.get(),
                                       weightTr.data(),
                                       twDesc.get(),
                                       totalWeightTr.data(),
                                       gradInputDesc.get(),
                                       gradInputRealTr.data()));
    if (dim > 2) {
        // NHWC -> NCHW and dealing with data type
        gradInputRealTr.reshape(inputContiguous.shape());
        gradInputTr.reshape(inputContiguous.shape());

        DiopiTensor gradInputTmpTr = gradInputTr;
        if (gradInputTr.dtype() != gradInputRealTr.dtype()) {
            gradInputTmpTr = requiresTensor(ctx, gradInputTr.shape(), gradInputRealTr.dtype());
        }

        DIOPI_CALL(cnnlTranspose(ctx, handle, gradInputRealTr, gradInputTmpTr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW));

        if (gradInputTmpTr.dtype() != gradInputTr.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, gradInputTr, gradInputTmpTr));
        }

    } else {
        DIOPI_CALL(diopiCopyInp(ctx, gradInputRealTr.tensorHandle(), gradInputTr.tensorHandle()));
    }

    return diopiSuccess;
}

diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                   diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    DiopiTensor inputTr(input);
    DiopiTensor targetTr(target);

    DIOPI_CHECK(labelSmoothing == 0, "Param label_smoothing is not supported by cnnl")
    DIOPI_CHECK(targetTr.dim() == inputTr.dim() - 1, "Probabilities for each class are not supported by cnnl");

    auto logTr = requiresTensor(ctx, inputTr.shape(), inputTr.dtype());
    DIOPI_CALL(diopiLogSoftmax(ctx, logTr.tensorHandle(), input, 1));
    DIOPI_CALL(diopiNLLLoss(ctx, out, logTr.tensorHandle(), target, weight, reduction, ignoreIndex));
    return diopiSuccess;
}
diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                           diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    DiopiTensor inputTr(input);
    DiopiTensor targetTr(target);
    DiopiTensor gradInputTr(gradInput);

    DIOPI_CHECK(labelSmoothing == 0, "param label_smoothing is not supported")
    DIOPI_CHECK(targetTr.dim() == inputTr.dim() - 1, "Probabilities for each class are not supported");

    auto logTr = requiresTensor(ctx, inputTr.shape(), inputTr.dtype());
    auto gradTmpTr = requiresTensor(ctx, gradInputTr.shape(), gradInputTr.dtype());

    DIOPI_CALL(diopiLogSoftmax(ctx, logTr.tensorHandle(), input, 1));
    // for nll loss backward, `input` should be logsoftmax out.
    DIOPI_CALL(diopiNLLLossBackward(ctx, gradTmpTr.tensorHandle(), gradOutput, logTr.tensorHandle(), target, weight, reduction, ignoreIndex));
    // for softmax backward, `output` should be logsoftmax out
    DIOPI_CALL(diopiLogSoftmaxBackward(ctx, gradInput, gradTmpTr.tensorHandle(), logTr.tensorHandle(), 1));
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

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
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
