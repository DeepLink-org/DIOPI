/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AscendTensor inputTensor(input);
    AscendTensor outTensor(out);
    if (out == input || out == nullptr || input == nullptr || inputTensor.numel() == 0 || outTensor.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnCast, ctx, input, outTensor.dtype(), out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
