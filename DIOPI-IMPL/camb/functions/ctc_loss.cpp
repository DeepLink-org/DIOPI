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
 */

static diopiError_t convertCTCLossReduction(cnnlCTCLossReduceMode_t *ctclossReduction , const diopiReduction_t reduction) {
    switch (reduction) {
        case ReductionNone:
            *ctclossReduction = CNNL_REDUCE_MODE_NONE;
        case ReductionMean:
            *ctclossReduction = CNNL_REDUCE_MODE_MEAN_BY_LABEL_LENGTH_AND_BATCH;
        case ReductionSum:
            *ctclossReduction = CNNL_REDUCE_MODE_SUM;
        default:
            // diopiReduction_t need update to support more mean types for ctc loss. 
            // CNNL_REDUCE_MODE_MEAN_BY_INPUT_LENGTHS
            // CNNL_REDUCE_MODE_MEAN_BY_LABEL_LENGTH_AND_BATCH
            DIOPI_CHECK(false, "The reduction mode does not supported.");
            return diopiError;
    }
    return diopiSuccess;
}

static diopiError_t ctcLossInternal(diopiContextHandle_t ctx, ) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);


    DIOPI_CALLCNNL(cnnlStatus_t cnnlCTCLoss(handle, 
                         const cnnlCTCLossDescriptor_t ctc_loss_desc, 
                         
                         const cnnlTensorDescriptor_t input_desc, 
                         const void *input, 
                         const cnnlTensorDescriptor_t labels_desc, 
                         const void *labels, 
                         const cnnlTensorDescriptor_t input_lengths_desc, 
                         const void *input_lengths,
                         const cnnlTensorDescriptor_t label_lengths_desc, 
                         const void *label_lengths, 
                         
                         void *workspace, 
                         size_t workspace_size, 
                         
                         const cnnlTensorDescriptor_t loss_desc, 
                         void *loss, 
                         const cnnlTensorDescriptor_t grads_desc, 
                         void *grads));
    return diopiSuccess;
}
DIOPI_API diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t negLogLikelihood, diopiTensorHandle_t logAlpha,
                                    diopiConstTensorHandle_t logProbs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t inputLengths,
                                    diopiConstTensorHandle_t targetLengths, int64_t blank, diopiReduction_t reduction, bool zeroInfinity) {
    // 
    
    // CNNL_SOFTMAX_NORMALIZATION || CNNL_LOG_SOFTMAX_NORMALIZATION
    cnnlCTCLossNormalizationMode_t ctcLossNormMode = CNNL_NONE_NORMALIZATION;
    cnnlCTCLossReduceMode_t ctcLossReduceMode;
    DIOPI_CALL(convertCTCLossReduction(&ctcLossReduceMode, reduction));
    cnnlCTCLossZeroInfinityMode_t ctcLossZeroInfMode = zeroInfinity ? CNNL_ZERO_INFINITY : CNNL_NONE_ZERO_INFINITY;

    CnnlResourceGuard<cnnlCTCLossDescriptor_t, cnnlCreateCTCLossDescriptor, cnnlDestroyCTCLossDescriptor> ctclossDescObj;
    cnnlCTCLossDescriptor_t ctcLossDesc = ctcLossDescObj.get();



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
