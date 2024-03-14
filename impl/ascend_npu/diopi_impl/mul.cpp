/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other);
    if (!inputAt.defined() || inputAt.numel() == 0 || !otherAt.defined() || otherAt.numel() == 0) {
        return diopiSuccess;
    }

    BEGIN_CALL_ACL_OP(out);
    // op_api::mul_out(inputAt, otherAt, outAt);
    EXEC_NPU_CMD(aclnnMul, inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other);
    if (!inputAt.defined() || inputAt.numel() == 0 || !otherAt.defined() || otherAt.numel() == 0) {
        return diopiSuccess;
    }

    // op_api::mul_(inputAt, otherAt);
    EXEC_NPU_CMD(aclnnInplaceMul, inputAt, otherAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, other);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    BEGIN_CALL_ACL_OP(out);
    EXEC_NPU_CMD(aclnnMuls, inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, other);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    // op_api::mul_(inputAt, otherAt);
    EXEC_NPU_CMD(aclnnInplaceMuls, inputAt, otherAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
