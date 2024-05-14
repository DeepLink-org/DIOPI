/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdd, ctx, input, other, alpha, out);
    return diopiSuccess;
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAdd, ctx, input, other, alpha);
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdds, ctx, input, other, alpha, out);
    return diopiSuccess;
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAdds, ctx, input, other, alpha);
    return diopiSuccess;
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSub, ctx, input, other, alpha, out);
    return diopiSuccess;
}

diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceSub, ctx, input, other, alpha);
    return diopiSuccess;
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSubs, ctx, input, other, alpha, out);
    return diopiSuccess;
}

diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceSubs, ctx, input, other, alpha);
    return diopiSuccess;
}

diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      diopiRoundMode_t roundingMode) {
    if (roundingMode == diopiRoundMode_t::RoundModeNone) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnDiv, ctx, input, other, out);
    } else {
        int mode = static_cast<int>(roundingMode);
        DIOPI_ASCEND_CALL_ACLNN(aclnnDivMod, ctx, input, other, mode, out);
    }
    return diopiSuccess;
}

diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t roundingMode) {
    if (roundingMode == diopiRoundMode_t::RoundModeNone) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceDiv, ctx, input, other);
    } else {
        int mode = static_cast<int>(roundingMode);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceDivMod, ctx, input, other, mode);
    }
    return diopiSuccess;
}

diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            diopiRoundMode_t roundingMode) {
    if (roundingMode == diopiRoundMode_t::RoundModeNone) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnDivs, ctx, input, other, out);
    } else {
        int mode = static_cast<int>(roundingMode);
        DIOPI_ASCEND_CALL_ACLNN(aclnnDivMods, ctx, input, other, mode, out);
    }
    return diopiSuccess;
}

diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t roundingMode) {
    if (roundingMode == diopiRoundMode_t::RoundModeNone) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceDivs, ctx, input, other);
    } else {
        int mode = static_cast<int>(roundingMode);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceDivMods, ctx, input, other, mode);
    }
    return diopiSuccess;
}

diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnMaximum, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnMinimum, ctx, input, other, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
