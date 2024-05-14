/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderTensorTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderTensorScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderScalarTensor, ctx, input, other, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
