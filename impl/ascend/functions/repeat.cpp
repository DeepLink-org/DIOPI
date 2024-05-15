/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeatsSize) {
    AscendTensor inputAt(input);
    // When repeatSize.len is equal to 0, out is the same as input.
    if (repeatsSize.len == 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, out, input);
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnRepeat, ctx, input, repeatsSize, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
