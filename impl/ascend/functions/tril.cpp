/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnTril, ctx, input, diagonal, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
