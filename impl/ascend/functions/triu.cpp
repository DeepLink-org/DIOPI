/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnTriu, ctx, input, diagonal, out);
    return diopiSuccess;
}

diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceTriu, ctx, input, diagonal);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
