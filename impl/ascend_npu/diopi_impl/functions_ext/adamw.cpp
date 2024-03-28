/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t expAvg, diopiTensorHandle_t expAvgSq,
                        diopiTensorHandle_t maxExpAvgSq, float lr, float beta1, float beta2, float eps, float weightDecay, int64_t step, bool amsgrad) {
    BEGIN_CALL_ACL_OP(input, grad, expAvg, expAvgSq, maxExpAvgSq);

    // maximize is not supported in diopi for now
    bool maximize = false;
    auto stepAt = at_npu::native::OpPreparation::apply_tensor_without_format({1}, inputAt.options().dtype(at::kLong));
    op_api::fill_(stepAt, step);

    // maxExpAvgSqAt is optional when amsgrad is false
    if (amsgrad) {
        EXEC_NPU_CMD(aclnnApplyAdamWV2, inputAt, expAvgAt, expAvgSqAt, maxExpAvgSqAt, gradAt, stepAt, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize);
    } else {
        c10::optional<at::Tensor> nullMaxExp;
        EXEC_NPU_CMD(aclnnApplyAdamWV2, inputAt, expAvgAt, expAvgSqAt, nullMaxExp, gradAt, stepAt, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize);
    }

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
