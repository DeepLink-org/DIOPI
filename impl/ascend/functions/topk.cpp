/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                       int64_t dim, bool largest, bool sorted) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnTopk, ctx, input, k, dim, largest, sorted, values, indices);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
