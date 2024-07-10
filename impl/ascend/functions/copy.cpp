
/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    AscendTensor srcAt(src);
    AscendTensor destAt(dest);
    if (src == nullptr || dest == nullptr || !srcAt.defined() || !destAt.defined() || srcAt.numel() <= 0 || destAt.numel() <= 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, dest, src);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
