/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    int64_t inputNumel = 0;
    int64_t otherNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    diopiGetTensorNumel(other, &otherNumel);
    if (input == nullptr || inputNumel == 0 || other == nullptr || otherNumel == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderTensorTensorf, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderTensorScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
    int64_t otherNumel = 0;
    diopiGetTensorNumel(other, &otherNumel);
    if (other == nullptr || otherNumel == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderScalarTensor, ctx, input, other, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
