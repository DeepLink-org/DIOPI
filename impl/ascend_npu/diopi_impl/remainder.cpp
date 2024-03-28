/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    if (!outAt.defined() || outAt.numel() == 0 || !inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::remainder_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    if (!outAt.defined() || outAt.numel() == 0 || !inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::remainder_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    if (!outAt.defined() || outAt.numel() == 0) {
        return diopiSuccess;
    }
    EXEC_NPU_CMD(aclnnRemainderScalarTensor, inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
