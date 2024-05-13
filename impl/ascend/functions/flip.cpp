/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnFlip, ctx, input, dims, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
