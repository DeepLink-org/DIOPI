/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */


#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

//namespace OP_IMPL_NS {
extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *value) {
    BEGIN_CALL_ACL_OP(input, value);
    if (inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    acl_op::fill_(inputAt, valueAt);
    END_CALL_ACL_OP();
}

}  // OP_IMPL_NS