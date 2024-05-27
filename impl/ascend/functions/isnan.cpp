/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiIsNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    // TODO: Waiting for Ascend to provide the aclnn kernel of isnan op.
    DIOPI_ASCEND_CALL_ACLNN(aclnnNeTensor, ctx, input, input, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
