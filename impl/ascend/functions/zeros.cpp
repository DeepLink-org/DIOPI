/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiZeros(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiSize_t size) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, out);
    return diopiSuccess;
}

diopiError_t diopiZeroInp(diopiContextHandle_t ctx, diopiTensorHandle_t self) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, self);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
