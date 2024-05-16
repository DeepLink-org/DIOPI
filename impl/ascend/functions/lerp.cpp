/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiLerpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                             diopiConstTensorHandle_t weight) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLerp, ctx, input, end, weight, out);
    return diopiSuccess;
}

diopiError_t diopiLerpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                             const diopiScalar_t* weight) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLerps, ctx, input, end, weight, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
