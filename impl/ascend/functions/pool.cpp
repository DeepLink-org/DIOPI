/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <algorithm>

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t outputSize) {
    // the shape of input: (N, C, H, W) or (C, H, W)
    // the shape of output: (N, C, S0, S1) or (C, S0, S1), where S = output_size

    AscendTensor inputAt(input);
    AscendTensor outAt(out);
    ASCEND_CHECK_ABORT((inputAt.dim() == 3 || inputAt.dim() == 4), "The input of adaptive_avg_pool2d should be 3_dimensional or 4_dimensional.");

    if ((outputSize.len == 1 && outputSize.data[0] == 0) || (outputSize.len == 2 && outputSize.data[0] == 0 && outputSize.data[1] == 0)) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnAdaptiveAvgPool2d, ctx, input, outputSize, out);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdaptiveAvgPool2dBackward, ctx, gradOutput, input, gradInput);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
