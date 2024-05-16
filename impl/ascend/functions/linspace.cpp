/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLinspace, ctx, start, end, steps, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
