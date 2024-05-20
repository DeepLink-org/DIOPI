/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    int8_t cubeMathType = 0;
    DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, input, other, out, cubeMathType);
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
