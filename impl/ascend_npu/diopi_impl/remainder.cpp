/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

//namespace OP_IMPL_NS {
extern "C" {

//at::Tensor & remainder_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out);
diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    acl_op::remainder_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

//at::Tensor & remainder_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    acl_op::remainder_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

//at::Tensor remainder(const at::Scalar & self, const at::Tensor & other);
diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    acl_op::remainder(inputAt, otherAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
