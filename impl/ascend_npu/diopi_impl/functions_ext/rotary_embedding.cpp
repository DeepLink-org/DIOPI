/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                            diopiConstTensorHandle_t sin, const bool conj, const bool interleaved) {
    BEGIN_CALL_ACL_OP(out, x, cos, sin);
    at::Tensor cosRepeated = acl_op::repeat(cosAt, {1, 1, 1, 2});
    at::Tensor sinRepeated = acl_op::repeat(sinAt, {1, 1, 1, 2});
    if (conj) {
        acl_op::neg_(sinRepeated);
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("RotaryMul").Input(xAt).Input(cosRepeated).Input(sinRepeated).Output(outAt).Run();
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
