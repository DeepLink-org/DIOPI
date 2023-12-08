/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

// at::Tensor & remainder_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out);
diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    if (!outAt.defined() || outAt.numel() <= 0 || !inputAt.defined() || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    acl_op::remainder_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

// at::Tensor & remainder_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    if (!outAt.defined() || outAt.numel() <= 0 || !inputAt.defined() || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    acl_op::remainder_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

// at::Tensor remainder(const at::Scalar & self, const at::Tensor & other);
diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    if (!outAt.defined() || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    acl_op::remainder_out(at::scalar_to_tensor(inputAt).to(outAt.scalar_type()), otherAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
