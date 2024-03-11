/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiLerpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                             diopiConstTensorHandle_t weight) {
    BEGIN_CALL_ACL_OP(input, end, weight, out)
    op_api::lerp_out(inputAt, endAt, weightAt, outAt);
    END_CALL_ACL_OP()
}

diopiError_t diopiLerpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                             const diopiScalar_t* weight) {
    BEGIN_CALL_ACL_OP(input, end, weight, out)
    op_api::lerp_out(inputAt, endAt, weightAt, outAt);
    END_CALL_ACL_OP()
}

}  // namespace OP_IMPL_NS
