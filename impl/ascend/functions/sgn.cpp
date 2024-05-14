/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSgn(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSign, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiSgnInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSign, ctx, input, input);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
