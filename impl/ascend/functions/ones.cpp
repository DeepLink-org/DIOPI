/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiOnes(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiSize_t size) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
