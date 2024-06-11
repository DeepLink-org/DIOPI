/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t grad, diopiTensorHandle_t expAvg,
                        diopiTensorHandle_t expAvgSq, diopiTensorHandle_t maxExpAvgSq, float lr, float beta1, float beta2, float eps, float weightDecay,
                        int64_t step, bool amsgrad) {
    // maximize is not supported in diopi for now
    bool maximize = false;
    // dtype of step supports int64、float32
    diopiScalar_t stepScalar = constructDiopiScalarT(diopi_dtype_float32, step);
    AscendTensor stepTensor;
    makeTensorFromScalar(ctx, stepTensor, &stepScalar);

    // maxExpAvgSq is optional when amsgrad is false
    if (amsgrad) {
        DIOPI_ASCEND_CALL_ACLNN(
            aclnnApplyAdamWV2, ctx, input, expAvg, expAvgSq, maxExpAvgSq, grad, stepTensor, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize);
    } else {
        diopiTensorHandle_t nullMaxExpAvgSq = nullptr;
        DIOPI_ASCEND_CALL_ACLNN(
            aclnnApplyAdamWV2, ctx, input, expAvg, expAvgSq, nullMaxExpAvgSq, grad, stepTensor, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize);
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
