/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    op_api::reciprocal_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    op_api::reciprocal_(inputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
