/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t runningMean,
                            diopiTensorHandle_t runningVar, bool training, double momentum, double eps) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnBatchNorm, ctx, input, weight, bias, runningMean, runningVar, training, momentum, eps, out, saveMean, saveInvstd);
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t runningMean, diopiConstTensorHandle_t runningVar, diopiConstTensorHandle_t saveMean,
                                    diopiConstTensorHandle_t saveInvstd, bool training, double eps) {
    int64_t gradMask[3] = {true, true, true};
    diopiSize_t gradMaskArray{gradMask, 3};
    DIOPI_ASCEND_CALL_ACLNN(aclnnBatchNormBackward,
                            ctx,
                            gradOutput,
                            input,
                            weight,
                            runningMean,
                            runningVar,
                            saveMean,
                            saveInvstd,
                            training,
                            eps,
                            gradMaskArray,
                            gradInput,
                            gradWeight,
                            gradBias);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
