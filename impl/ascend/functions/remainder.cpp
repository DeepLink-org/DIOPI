/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstdint>

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    if (input == nullptr || out == nullptr) {
        return diopiSuccess;
    }
    int64_t inputNumel = 0;
    int64_t outNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    diopiGetTensorNumel(out, &outNumel);
    if (inputNumel == 0 || outNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderTensorTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    if (input == nullptr || out == nullptr) {
        return diopiSuccess;
    }
    int64_t inputNumel = 0;
    int64_t outNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    diopiGetTensorNumel(out, &outNumel);
    if (inputNumel == 0 || outNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderTensorScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
    if (out == nullptr) {
        return diopiSuccess;
    }
    int64_t outNumel = 0;
    diopiGetTensorNumel(out, &outNumel);
    if (outNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderScalarTensor, ctx, input, other, out);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
