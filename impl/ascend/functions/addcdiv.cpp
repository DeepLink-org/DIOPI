/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (inputNumel != 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnAddcdiv, ctx, input, tensor1, tensor2, value, out);
    }
    return diopiSuccess;
}

diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (inputNumel != 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAddcdiv, ctx, input, tensor1, tensor2, value);
    }
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
