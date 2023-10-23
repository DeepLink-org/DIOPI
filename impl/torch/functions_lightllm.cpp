/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <diopi/functions_lightllm.h>
#include <torch/nn.h>
#include <torch/optim.h>

#include "context.h"
#include "helper.hpp"

extern "C" {

diopiError_t diopiDestindexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
    return diopiSuccess;
}

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t AttentionOut, diopiConstTensorHandle_t Q, diopiConstTensorHandle_t K,
                                          diopiConstTensorHandle_t BLoc, diopiConstTensorHandle_t BStartLoc, diopiConstTensorHandle_t BSeqlen,
                                          int maxInputLen) {
    return diopiSuccess;
}

diopiError_t diopiTokenSoftmaxInference(diopiContextHandle_t ctx, diopiTensorHandle_t ProbOut, diopiConstTensorHandle_t Logics,
                                        diopiConstTensorHandle_t BStartLoc, diopiConstTensorHandle_t BSeqlen, int maxInputLen) {
    return diopiSuccess;
}

diopiError_t diopiTokenReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t Out, diopiConstTensorHandle_t Prob, diopiConstTensorHandle_t V,
                                        diopiConstTensorHandle_t BLoc, diopiConstTensorHandle_t BStartLoc, diopiConstTensorHandle_t BSeqlen, int maxInputLen) {
    return diopiSuccess;
}

diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t Out, diopiConstTensorHandle_t Logics, diopiConstTensorHandle_t V,
                                               diopiConstTensorHandle_t BLoc, diopiConstTensorHandle_t BStartLoc, diopiConstTensorHandle_t BSeqlen,
                                               int maxInputLen, diopiConstTensorHandle_t otherKVIndex) {
    return diopiSuccess;
}

}  // extern "C"
