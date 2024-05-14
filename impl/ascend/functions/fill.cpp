/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, input, value);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
