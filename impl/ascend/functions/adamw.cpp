/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t grad, diopiTensorHandle_t expAvg,
                        diopiTensorHandle_t expAvgSq, diopiTensorHandle_t maxExpAvgSq, float lr, float beta1, float beta2, float eps, float weightDecay,
                        int64_t step, bool amsgrad) {
    // maximize is not supported in diopi for now
    bool maximize = false;
    DIOPI_ASCEND_CALL_ACLNN(aclnnApplyAdamWV2, ctx, input, expAvg, expAvgSq, maxExpAvgSq, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize);
}

}  // namespace ascend
}  // namespace impl
