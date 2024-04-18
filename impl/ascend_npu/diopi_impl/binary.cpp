/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

std::string roundModeStr(diopiRoundMode_t roundMode) {
    if (roundMode == diopiRoundMode_t::RoundModeFloor) {
        return "floor";
    } else if (roundMode == diopiRoundMode_t::RoundModeTrunc) {
        return "trunc";
    } else {
        return "";
    }
    return "";
}

diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      diopiRoundMode_t roundingMode) {
    BEGIN_CALL_ACL_OP(input, other, out);
    if (roundingMode == diopiRoundMode_t::RoundModeNone) {
        EXEC_NPU_CMD(aclnnDiv, inputAt, otherAt, outAt);
    } else {
        int mode = static_cast<int>(roundingMode);  // the mode of aclnn is matched with of diopiRoundMode_t
        EXEC_NPU_CMD(aclnnDivMod, inputAt, otherAt, mode, outAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t roundingMode) {
    BEGIN_CALL_ACL_OP(input, other);
    if (roundingMode == diopiRoundMode_t::RoundModeNone) {
        EXEC_NPU_CMD(aclnnInplaceDiv, inputAt, otherAt);
    } else {
        int mode = static_cast<int>(roundingMode);  // the mode of aclnn is matched with of diopiRoundMode_t
        EXEC_NPU_CMD(aclnnInplaceDivMod, inputAt, otherAt, mode);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            diopiRoundMode_t roundingMode) {
    BEGIN_CALL_ACL_OP(input, other, out);
    if (roundingMode == diopiRoundMode_t::RoundModeNone) {
        EXEC_NPU_CMD(aclnnDivs, inputAt, otherAt, outAt);
    } else {
        int mode = static_cast<int>(roundingMode);  // the mode of aclnn is matched with of diopiRoundMode_t
        EXEC_NPU_CMD(aclnnDivMods, inputAt, otherAt, mode, outAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t roundingMode) {
    BEGIN_CALL_ACL_OP(input, other);
    if (roundingMode == diopiRoundMode_t::RoundModeNone) {
        EXEC_NPU_CMD(aclnnInplaceDivs, inputAt, otherAt);
    } else {
        int mode = static_cast<int>(roundingMode);  // the mode of aclnn is matched with of diopiRoundMode_t
        EXEC_NPU_CMD(aclnnInplaceDivMods, inputAt, otherAt, mode);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, out, alpha, other);
    EXEC_NPU_CMD(aclnnAdd, inputAt, otherAt, alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha);
    if (otherAt.is_cpu()) {
        otherAt = otherAt.to(inputAt.device());
    }
    EXEC_NPU_CMD(aclnnInplaceAdd, inputAt, otherAt, alphaAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha, out);
    EXEC_NPU_CMD(aclnnAdds, inputAt, otherAt, alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha);
    EXEC_NPU_CMD(aclnnInplaceAdds, inputAt, otherAt, alphaAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, out, alpha, other);
    if (!outAt.defined() || outAt.numel() <= 0) {
        return diopiSuccess;
    }

    EXEC_NPU_CMD(aclnnSub, inputAt, otherAt, alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha);
    if (!inputAt.defined() || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    EXEC_NPU_CMD(aclnnInplaceSub, inputAt, otherAt, alphaAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha, out);
    if (!outAt.defined() || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    EXEC_NPU_CMD(aclnnSubs, inputAt, otherAt, alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha);
    if (!inputAt.defined() || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    EXEC_NPU_CMD(aclnnInplaceSubs, inputAt, otherAt, alphaAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
