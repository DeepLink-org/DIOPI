/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

//namespace OP_IMPL_NS {
extern "C" {

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, out, alpha, other);
    acl_op::add_out(inputAt, otherAt, alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha);
    acl_op::add_out(inputAt, otherAt, alphaAt, inputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha, out);
    acl_op::add_out(inputAt, at::scalar_to_tensor(otherAt).to(inputAt.dtype()), alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, other, alpha);
    acl_op::add_(inputAt, at::scalar_to_tensor(otherAt).to(inputAt.dtype()), alphaAt);
    END_CALL_ACL_OP();
}

}  // OP_IMPL_NS
