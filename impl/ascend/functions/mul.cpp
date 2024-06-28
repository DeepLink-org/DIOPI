/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    AscendTensor inputAt(input);
    AscendTensor otherAt(other);
    if (!inputAt.defined() || inputAt.numel() == 0 || !otherAt.defined() || otherAt.numel() == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    AscendTensor inputAt(input);
    AscendTensor otherAt(other);
    if (!inputAt.defined() || inputAt.numel() == 0 || !otherAt.defined() || otherAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMul, ctx, input, other);
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    AscendTensor inputAt(input);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnMuls, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    AscendTensor inputAt(input);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMuls, ctx, input, other);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
