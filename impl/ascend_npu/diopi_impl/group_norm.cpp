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
    int64_t N = inputAt.sizes()[0];
    int64_t C = inputAt.sizes()[1];
    int64_t HW = inputAt.numel() / (N * C);
    eps = (eps < 1e-5) ? 1e-5 : eps;
    EXEC_NPU_CMD(aclnnGroupNorm, inputAt, weightAt, biasAt, N, C, HW, numGroups, eps, outAt, saveMeanAt, saveInvstdAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
