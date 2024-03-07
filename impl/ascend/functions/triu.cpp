/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnTriu, ctx, inputAt, diagonal, outAt);
    return diopiSuccess;
}

diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
    AscendTensor inputAt(input);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceTriu, ctx, inputAt, diagonal);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
