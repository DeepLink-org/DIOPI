/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

extern "C" {
diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeatSize) {
    BEGIN_CALL_ACL_OP(out, input, repeatSize);
    if (repeatSizeAt.empty()) {
        outAt.copy_(inputAt);
        return diopiSuccess;
    }
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    op_api::repeat(inputAt, repeatSizeAt);
    END_CALL_ACL_OP();
}

}  // extern C
