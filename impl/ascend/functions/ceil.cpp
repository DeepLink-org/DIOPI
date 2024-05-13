/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiCeil(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnCeil, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiCeilInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCeil, ctx, input);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
