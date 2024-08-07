/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, input, dim, index, out);
    return diopiSuccess;
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t inputSizes,
                                      int64_t dim, diopiConstTensorHandle_t index) {
    AscendTensor gradInputTensor(gradInput);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradInput);
    diopiScalar_t one = constructDiopiScalarT(gradInputTensor.dtype(), 1);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexAdd, ctx, gradInput, dim, index, gradOutput, &one, gradInput);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
