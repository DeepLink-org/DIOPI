/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    AscendTensor inputAt(input);
    AscendTensor mat2At(mat2);
    AscendTensor outAt(out);

    if (inputAt.numel() == 0 || mat2At.numel() == 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, out);
        return diopiSuccess;
    }

    int cubeMathType = 0;
    DIOPI_ASCEND_CALL_ACLNN(aclnnMm, ctx, input, mat2, out, cubeMathType);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
