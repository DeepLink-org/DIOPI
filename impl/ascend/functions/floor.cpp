/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnFloor, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFloor, ctx, input);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
