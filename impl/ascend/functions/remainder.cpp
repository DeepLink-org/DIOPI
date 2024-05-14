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
    AscendTensor inputAt(input);
    AscendTensor otherAt(other);
    AscendTensor outAt(out);

    diopiDtype_t promotedType = promoteTypes(inputAt.dtype(), otherAt.dtype());
    if (input == nullptr || inputAt.numel() == 0 || other == nullptr || otherAt.numel() == 0) {
        return diopiSuccess;
    }

    castTensor(ctx, inputAt, promotedType);
    castTensor(ctx, otherAt, promotedType);
    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderTensorTensor, ctx, inputAt, otherAt, outAt);
    return diopiSuccess;
}

diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    diopiDtype_t promotedType = promoteTypes(inputAt.dtype(), other->stype);
    castTensor(ctx, inputAt, promotedType);
    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderTensorScalar, ctx, inputAt, other, outAt);
    return diopiSuccess;
}

diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
    AscendTensor otherAt(other);
    AscendTensor outAt(out);
    if (other == nullptr || otherAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnRemainderScalarTensor, ctx, input, otherAt, outAt);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
