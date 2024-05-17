/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    AscendTensor inAt(input);
    AscendTensor matAt(mat2);
    ASCEND_CHECK_ABORT(inAt.dtype() == matAt.dtype(), "[diopiBmm] tensors dtype does not matched.");

    int cubeMathType = 0;
    DIOPI_ASCEND_CALL_ACLNN(aclnnBatchMatMul, ctx, input, mat2, out, cubeMathType);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
