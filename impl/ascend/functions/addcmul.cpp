/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    AscendTensor inputAt(input);
    if (inputAt.numel() != 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnAddcmul, ctx, inputAt, tensor1, tensor2, value, out);
    }
    return diopiSuccess;
}

diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    AscendTensor inputAt(input);
    if (inputAt.numel() != 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAddcmul, ctx, input, tensor1, tensor2, value);
    }

    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
