/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                             diopiConstTensorHandle_t value) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    // step1 copy inputAt to outAt
    if (outAt.data() != inputAt.data()) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, outAt, inputAt);
    }
    // step2 call aclnnInplaceMaskedFillTensor on outAt
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMaskedFillTensor, ctx, outAt, mask, value);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMaskedFillTensor, ctx, inputAt, mask, value);
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                   const diopiScalar_t* value) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    // step1 copy inputAt to outAt
    if (outAt.data() != inputAt.data()) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, outAt, inputAt);
    }
    // step2 call aclnnInplaceMaskedFillScalar on outAt
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMaskedFillScalar, ctx, outAt, mask, value);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMaskedFillScalar, ctx, inputAt, mask, value);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
