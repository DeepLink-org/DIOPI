/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                        diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    int8_t cubeMathType = 0;
    DIOPI_ASCEND_CALL_ACLNN(aclnnAddmm, ctx, input, mat1, mat2, beta, alpha, out, cubeMathType);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
