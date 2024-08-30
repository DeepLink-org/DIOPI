/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBatchNormStats(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input, double eps) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBatchNormStats, ctx, input, eps, mean, invstd);
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackwardReduce(diopiContextHandle_t ctx, diopiTensorHandle_t sumDy, diopiTensorHandle_t sumDyXmu, diopiTensorHandle_t gradWeight,
                                          diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, diopiConstTensorHandle_t weight, bool inputG,
                                          bool weightG, bool biasG) {
    DIOPI_ASCEND_CALL_ACLNN(
        aclnnBatchNormReduceBackward, ctx, gradOut, input, mean, invstd, weight, inputG, weightG, biasG, sumDy, sumDyXmu, gradWeight, gradBias);
    return diopiSuccess;
}

diopiError_t diopiBatchNormGatherStatsWithCounts(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input,
                                                 diopiConstTensorHandle_t meanAll, diopiConstTensorHandle_t invstdAll, diopiTensorHandle_t runningMean,
                                                 diopiTensorHandle_t runningVar, float momentum, float eps, diopiConstTensorHandle_t counts) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBatchNormGatherStatsWithCounts, ctx, input, meanAll, invstdAll, runningMean, runningVar, momentum, eps, counts, mean, invstd);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
