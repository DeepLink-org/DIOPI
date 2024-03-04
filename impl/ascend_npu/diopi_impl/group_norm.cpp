/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t numGroups, double eps) {
    BEGIN_CALL_ACL_OP(input, weight, bias, out, saveMean, saveInvstd);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }
    int64_t n = inputAt.sizes()[0];
    int64_t c = inputAt.sizes()[1];
    int64_t hw = inputAt.numel() / (n * c);
    eps = (eps < 1e-5) ? 1e-5 : eps;
    EXEC_NPU_CMD(aclnnGroupNorm, inputAt, weightAt, biasAt, n, c, hw, numGroups, eps, outAt, saveMeanAt, saveInvstdAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, int64_t numGroups) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradBias, gradOutput, input, weight, mean, rstd);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }
    int64_t n = inputAt.sizes()[0];
    int64_t c = inputAt.sizes()[1];
    int64_t hw = inputAt.numel() / (n * c);
    std::array<bool, 3> gradInputMask = {true, true, true};
    EXEC_NPU_CMD(
        aclnnGroupNormBackward, gradOutputAt, inputAt, meanAt, rstdAt, weightAt, n, c, hw, numGroups, gradInputMask, gradInputAt, gradWeightAt, gradBiasAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
