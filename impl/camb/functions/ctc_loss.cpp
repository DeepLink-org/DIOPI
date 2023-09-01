/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/debug.hpp"

namespace impl {
namespace camb {
/**
 * @brief Computes the Connectionist Temporal Classification loss.
 *
 */
static diopiError_t convertCTCLossReduction(cnnlCTCLossReduceMode_t *ctclossReduction, const diopiReduction_t reduction) {
    switch (reduction) {
        case ReductionNone:
            *ctclossReduction = CNNL_REDUCE_MODE_NONE;
            break;
        case ReductionMean:
            *ctclossReduction = CNNL_REDUCE_MODE_MEAN_BY_LABEL_LENGTH_AND_BATCH;
            break;
        case ReductionSum:
            *ctclossReduction = CNNL_REDUCE_MODE_SUM;
            break;
        default:
            DIOPI_CHECK(false, "The reduction mode does not supported.");
    }
    return diopiSuccess;
}

static diopiError_t maxTensorElement(diopiContextHandle_t ctx, DiopiTensor targetLengthTensor, int32_t *maxTargetLen) {
    std::vector<int32_t> targetLengthVector(targetLengthTensor.numel(), 0);
    auto cnrtRet = cnrtMemcpyAsync(
        targetLengthVector.data(), targetLengthTensor.data(), sizeof(int32_t) * targetLengthTensor.numel(), getStream(ctx), cnrtMemcpyDevToHost);
    DIOPI_CHECK(cnrtRet == cnrtSuccess, "[diopiCTCLoss] Memory copy from Device to Host failed.");
    syncStreamInCtx(ctx);
    *maxTargetLen = *std::max_element(targetLengthVector.begin(), targetLengthVector.end());
    return diopiSuccess;
}

static diopiError_t ctcLoss(diopiContextHandle_t ctx, DiopiTensor lossTensor, DiopiTensor gradTensor, DiopiTensor logProbsTensor, DiopiTensor targetTensor,
                            DiopiTensor inputLengths, DiopiTensor targetLengths, cnnlCTCLossDescriptor_t ctcLossDesc, bool backward) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlTensorDesc lossTensorDesc(lossTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradTensorDesc(gradTensor, CNNL_LAYOUT_TNC);
    CnnlTensorDesc logProbsTensorDesc(logProbsTensor, CNNL_LAYOUT_TNC);
    CnnlTensorDesc targetTensorDesc(targetTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputLengthsDesc(inputLengths, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetLengthsDesc(targetLengths, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetCTCLossWorkspaceSize(handle, ctcLossDesc, logProbsTensorDesc.get(), backward, &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
        DIOPI_CHECK(workspace != nullptr, "[diopiCTCLoss] require buffers: size = %d, for workspace failed.", workspaceSize);
    }

    DIOPI_CALLCNNL(cnnlCTCLoss(handle,
                               ctcLossDesc,
                               logProbsTensorDesc.get(),
                               logProbsTensor.data(),
                               targetTensorDesc.get(),
                               targetTensor.data(),
                               inputLengthsDesc.get(),
                               inputLengths.data(),
                               targetLengthsDesc.get(),
                               targetLengths.data(),
                               workspace,
                               workspaceSize,
                               lossTensorDesc.get(),
                               lossTensor.data(),
                               backward ? gradTensorDesc.get() : nullptr,
                               backward ? gradTensor.data() : nullptr));
    return diopiSuccess;
}

diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t negLogLikelihood, diopiTensorHandle_t logAlpha,
                          diopiConstTensorHandle_t logProbs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t inputLengths,
                          diopiConstTensorHandle_t targetLengths, int64_t blank, diopiReduction_t reduction, bool zeroInfinity) {
    // input and nll, la, out.
    DiopiTensor outTensor(out);
    DiopiTensor negLLTensor(negLogLikelihood);
    DiopiTensor logAlphaTensor(logAlpha);
    DiopiTensor logProbsTensor(logProbs);

    std::vector<DiopiTensor *> inputTensorsVecPtr{&logProbsTensor};
    std::set<diopiDtype_t> inputSupportedDtype{diopi_dtype_float32, diopi_dtype_float64};
    DIOPI_CALL(autoCastTensorType(ctx, inputTensorsVecPtr, inputSupportedDtype));

    // lable and length.
    DiopiTensor inputLengthsTensor(inputLengths);
    DiopiTensor targetTensor(targets);
    DiopiTensor targetLengthTensor(targetLengths);

    std::vector<DiopiTensor *> labelLengthTensorsVecPtr{&inputLengthsTensor, &targetTensor, &targetLengthTensor};
    std::set<diopiDtype_t> labelLengthSupportedDtype{diopi_dtype_int32, diopi_dtype_int64};
    DIOPI_CALL(autoCastTensorType(ctx, labelLengthTensorsVecPtr, labelLengthSupportedDtype));

    DiopiTensor outTensorTemp = outTensor;
    if (outTensor.dtype() != logProbsTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTemp, logProbsTensor.dtype()));
    }

    // ctc_loss descriptor
    CnnlResourceGuard<cnnlCTCLossDescriptor_t, cnnlCreateCTCLossDescriptor, cnnlDestroyCTCLossDescriptor> ctcLossDescObj;
    cnnlCTCLossDescriptor_t ctcLossDesc = ctcLossDescObj.get();
    int32_t maxInputLength = logProbsTensor.shape()[0];
    int32_t maxTargetLen = 0;
    DIOPI_CALL(maxTensorElement(ctx, targetLengthTensor, &maxTargetLen));

    cnnlCTCLossNormalizationMode_t ctcLossNormMode = CNNL_LOG_SOFTMAX_NORMALIZATION;
    cnnlCTCLossReduceMode_t ctcLossReduceMode;
    DIOPI_CALL(convertCTCLossReduction(&ctcLossReduceMode, reduction));
    cnnlCTCLossZeroInfinityMode_t ctcLossZeroInfMode = zeroInfinity ? CNNL_ZERO_INFINITY : CNNL_NONE_ZERO_INFINITY;

    DIOPI_CALLCNNL(cnnlSetCTCLossDescriptor(ctcLossDesc, ctcLossNormMode, ctcLossReduceMode, ctcLossZeroInfMode, blank, maxInputLength, maxTargetLen));

    DiopiTensor gradTensor = requiresTensor(ctx, logProbsTensor.shape(), logProbsTensor.dtype());
    DIOPI_CALL(ctcLoss(ctx, outTensorTemp, gradTensor, logProbsTensor, targetTensor, inputLengthsTensor, targetLengthTensor, ctcLossDesc, false));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));

    return diopiSuccess;
}

diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                  diopiConstTensorHandle_t logProbs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t inputLengths,
                                  diopiConstTensorHandle_t targetLengths, diopiConstTensorHandle_t negLogLikelihood, diopiConstTensorHandle_t logAlpha,
                                  int64_t blank, diopiReduction_t reduction, bool zeroInfinity) {
    // input and nll, la, out.
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);

    DiopiTensor negLLTensor(negLogLikelihood);
    DiopiTensor logProbsTensor(logProbs);
    DiopiTensor logAlphaTensor(logAlpha);

    std::vector<DiopiTensor *> inputTensorsVecPtr{&gradOutputTensor, &logProbsTensor};
    std::set<diopiDtype_t> inputSupportedDtype{diopi_dtype_float32, diopi_dtype_float64};
    DIOPI_CALL(autoCastTensorType(ctx, inputTensorsVecPtr, inputSupportedDtype));

    // lable and length.
    DiopiTensor targetTensor(targets);
    DiopiTensor inputLengthsTensor(inputLengths);
    DiopiTensor targetLengthTensor(targetLengths);

    std::vector<DiopiTensor *> labelLengthTensorsVecPtr{&targetTensor, &inputLengthsTensor, &targetLengthTensor};
    std::set<diopiDtype_t> labelLengthSupportedDtype{diopi_dtype_int32, diopi_dtype_int64};
    DIOPI_CALL(autoCastTensorType(ctx, labelLengthTensorsVecPtr, labelLengthSupportedDtype));

    DiopiTensor gradInputTensorTemp = gradInputTensor;
    if (gradInputTensor.dtype() != logProbsTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensorTemp, logProbsTensor.dtype()));
    }

    // ctc_loss descriptor
    CnnlResourceGuard<cnnlCTCLossDescriptor_t, cnnlCreateCTCLossDescriptor, cnnlDestroyCTCLossDescriptor> ctcLossDescObj;
    cnnlCTCLossDescriptor_t ctcLossDesc = ctcLossDescObj.get();

    int32_t maxInputLength = logProbsTensor.shape()[0];
    int32_t maxTargetLen = 0;
    DIOPI_CALL(maxTensorElement(ctx, targetLengthTensor, &maxTargetLen));

    cnnlCTCLossNormalizationMode_t ctcLossNormMode = CNNL_LOG_SOFTMAX_NORMALIZATION;
    cnnlCTCLossReduceMode_t ctcLossReduceMode;
    DIOPI_CALL(convertCTCLossReduction(&ctcLossReduceMode, reduction));
    cnnlCTCLossZeroInfinityMode_t ctcLossZeroInfMode = zeroInfinity ? CNNL_ZERO_INFINITY : CNNL_NONE_ZERO_INFINITY;

    DIOPI_CALLCNNL(cnnlSetCTCLossDescriptor(ctcLossDesc, ctcLossNormMode, ctcLossReduceMode, ctcLossZeroInfMode, blank, maxInputLength, maxTargetLen));

    DiopiTensor lossTensor;
    if (ctcLossReduceMode == CNNL_REDUCE_MODE_NONE) {
        lossTensor = requiresTensor(ctx, {logProbsTensor.shape()[1]}, logProbsTensor.dtype());
    } else {
        lossTensor = requiresTensor(ctx, {1}, logProbsTensor.dtype());
    }
    DIOPI_CALL(ctcLoss(ctx, lossTensor, gradInputTensorTemp, logProbsTensor, targetTensor, inputLengthsTensor, targetLengthTensor, ctcLossDesc, true));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTemp));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
