/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiSize_t outSize;
    diopiGetTensorShape(out, &outSize);
    DIOPI_ASCEND_CALL_ACLNN(aclnnExpand, ctx, input, outSize, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
