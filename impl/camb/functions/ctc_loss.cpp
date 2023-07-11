/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {                         
/**
 * @brief Computes the Connectionist Temporal Classification loss.
 * 
 * 
 * 
 * 
 * 
 */

static diopiError_t convertCTCLossReduction(cnnlCTCLossReduceMode_t *ctclossReduction , const diopiReduction_t reduction) {
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
            // diopiReduction_t need update to support more mean types for ctc loss. 
            // CNNL_REDUCE_MODE_MEAN_BY_INPUT_LENGTHS
            // CNNL_REDUCE_MODE_MEAN_BY_LABEL_LENGTH_AND_BATCH
            DIOPI_CHECK(false, "The reduction mode does not supported.");
            return diopiErrorOccurred;
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t negLogLikelihood, diopiTensorHandle_t logAlpha,
                                    diopiConstTensorHandle_t logProbs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t inputLengths,
                                    diopiConstTensorHandle_t targetLengths, int64_t blank, diopiReduction_t reduction, bool zeroInfinity) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor logProbsTensor(logProbs);

    std::vector<DiopiTensor *> inputOutputTensorsVecPtr{&outTensor, &logProbsTensor};
    std::set<diopiDtype_t> inputOutputSupportedDtype{diopi_dtype_float32, diopi_dtype_float64};
    DIOPI_CALL(autoCastTensorType(ctx, inputOutputTensorsVecPtr, inputOutputSupportedDtype));
    outTensor = *inputOutputTensorsVecPtr[0];
    logProbsTensor = *inputOutputTensorsVecPtr[1];

    DiopiTensor inputLengthTensor(inputLengths);
    DiopiTensor targetsTensor(targets);
    DiopiTensor targetsLengthTensor(targetLengths);

    std::vector<DiopiTensor *> labelLengthTensorsVecPtr{&inputLengthTensor, &targetsTensor, &targetsLengthTensor};
    std::set<diopiDtype_t> labelLengthSupportedDtype{diopi_dtype_int32, diopi_dtype_int64};
    DIOPI_CALL(autoCastTensorType(ctx, labelLengthTensorsVecPtr, labelLengthSupportedDtype));
    inputLengthTensor = *labelLengthTensorsVecPtr[0];
    targetsTensor = *labelLengthTensorsVecPtr[1];
    targetsLengthTensor = *labelLengthTensorsVecPtr[2];

    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc logProbsDesc(logProbsTensor, CNNL_LAYOUT_TNC);
    CnnlTensorDesc targetsDesc(targetsTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputLengthDesc(inputLengthTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetsLengthDesc(targetsLengthTensor, CNNL_LAYOUT_ARRAY);
    // CnnlTensorDesc

    CnnlResourceGuard<cnnlCTCLossDescriptor_t, cnnlCreateCTCLossDescriptor, cnnlDestroyCTCLossDescriptor> ctcLossDescObj;
    cnnlCTCLossDescriptor_t ctcLossDesc = ctcLossDescObj.get();

    auto batchSize = logProbsTensor.shape()[1];
    auto numLabels = logProbsTensor.shape()[2];
    int maxInputLength = logProbsTensor.shape()[0];

    // for (decltype(batchSize) i = 0; i < batchSize; i++) {
    //     if (maxInputLength)
    // }
    std::cout << "here:" << std::endl;
    // int64_t maxLabelLength = 0;
    void *mll = malloc(sizeof(int32_t));
    
    diopiTensorHandle_t maxLabelLen = nullptr;
    std::vector<int64_t> shape{1};
    diopiSize_t size(shape.data(), 1);
    DIOPI_CALL(diopiRequireTensor(ctx, &maxLabelLen, &size, nullptr, diopi_dtype_int32, diopi_device));
    DIOPI_CALL(diopiMaxAll(ctx, maxLabelLen, targetLengths));

    DiopiTensor maxLabelLenTensor(maxLabelLen);
    printDevData(ctx, maxLabelLenTensor, "[max label len]");
    cnrtMemcpy(mll, maxLabelLenTensor.data(), sizeof(int32_t), cnrtMemcpyDevToHost);
    int32_t maxLabelLength = *reinterpret_cast<int32_t *>(mll);
    std::cout << "[Host Label Leng] " << maxLabelLength << std::endl;

    cnnlCTCLossNormalizationMode_t ctcLossNormMode = CNNL_LOG_SOFTMAX_NORMALIZATION;
    cnnlCTCLossReduceMode_t ctcLossReduceMode;
    DIOPI_CALL(convertCTCLossReduction(&ctcLossReduceMode, reduction));
    // cnnlCTCLossZeroInfinityMode_t ctcLossZeroInfMode = zeroInfinity ? CNNL_NONE_ZERO_INFINITY_PROBS_GRADS : CNNL_NONE_ZERO_INFINITY;
    cnnlCTCLossZeroInfinityMode_t ctcLossZeroInfMode = zeroInfinity ? CNNL_ZERO_INFINITY : CNNL_NONE_ZERO_INFINITY;

    DIOPI_CHECK(blank == 0, "ctc_loss only support blank = 0 on cambricon.");
    DIOPI_CALLCNNL(cnnlSetCTCLossDescriptor(ctcLossDesc, ctcLossNormMode, ctcLossReduceMode, ctcLossZeroInfMode, blank, maxInputLength, static_cast<int>(maxLabelLength)));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetCTCLossWorkspaceSize(handle, ctcLossDesc, logProbsDesc.get(), false, &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
        DIOPI_CHECK(workspace != nullptr, "[diopiCTCLoss] require buffers: size = %d, for workspace failed.", workspaceSize);
    }
    printDevData(ctx, outTensor, "[CTCLoss] loss");
    DIOPI_CALLCNNL(cnnlCTCLoss(handle,
                         ctcLossDesc,
                         logProbsDesc.get(),
                         logProbsTensor.data(),
                         targetsDesc.get(),
                         targetsTensor.data(), 
                         inputLengthDesc.get(),
                         inputLengthTensor.data(),
                         targetsLengthDesc.get(),
                         targetsLengthTensor.data(),
                         workspace,
                         workspaceSize,
                         outDesc.get(), //ctcLossDesc,//const cnnlTensorDescriptor_t loss_desc,
                         outTensor.data(),//void *loss,
                         nullptr,
                         nullptr));
    // CNNL_SOFTMAX_NORMALIZATION || CNNL_LOG_SOFTMAX_NORMALIZATION
    // cnnlCTCLossNormalizationMode_t ctcLossNormMode = CNNL_NONE_NORMALIZATION;
    // cnnlCTCLossReduceMode_t ctcLossReduceMode;
    // DIOPI_CALL(convertCTCLossReduction(&ctcLossReduceMode, reduction));
    // cnnlCTCLossZeroInfinityMode_t ctcLossZeroInfMode = zeroInfinity ? CNNL_ZERO_INFINITY : CNNL_NONE_ZERO_INFINITY;

    // CnnlResourceGuard<cnnlCTCLossDescriptor_t, cnnlCreateCTCLossDescriptor, cnnlDestroyCTCLossDescriptor> ctclossDescObj;
    // cnnlCTCLossDescriptor_t ctcLossDesc = ctcLossDescObj.get();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t logProbs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t inputLengths,
                                            diopiConstTensorHandle_t targetLengths, diopiConstTensorHandle_t negLogLikelihood,
                                            diopiConstTensorHandle_t logAlpha, int64_t blank, diopiReduction_t reduction, bool zeroInfinity) {
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
