/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {
diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);

    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnCumsum, ctx, inputAt, dim, outAt.dtype(), outAt);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
