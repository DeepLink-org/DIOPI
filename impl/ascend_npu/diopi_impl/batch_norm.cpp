/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t runningMean,
                            diopiTensorHandle_t runningVar, bool training, double momentum, double eps) {
    BEGIN_CALL_ACL_OP(out, saveMean, saveInvstd, input, weight, bias, runningMean, runningVar);
    op_api::native_batch_norm_out(inputAt, weightAt, biasAt, runningMeanAt, runningVarAt, training, momentum, eps, outAt, saveMeanAt, saveInvstdAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t runningMean, diopiConstTensorHandle_t runningVar, diopiConstTensorHandle_t saveMean,
                                    diopiConstTensorHandle_t saveInvstd, bool training, double eps) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradBias, gradOutput, input, weight, runningMean, runningVar, saveMean, saveInvstd);
    std::array<bool, 3> gradInputMask{gradInput != nullptr, gradWeight != nullptr, gradBias != nullptr};
    EXEC_NPU_CMD(aclnnBatchNormBackward,
                 gradOutputAt,
                 inputAt,
                 weightAt,
                 runningMeanAt,
                 runningVarAt,
                 saveMeanAt,
                 saveInvstdAt,
                 training,
                 eps,
                 gradInputMask,
                 gradInputAt,
                 gradWeightAt,
                 gradBiasAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
