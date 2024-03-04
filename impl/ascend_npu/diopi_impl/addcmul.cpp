/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    BEGIN_CALL_ACL_OP(input, tensor1, tensor2, value, out);
    op_api::addcmul_out(inputAt, tensor1At, tensor2At, valueAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    BEGIN_CALL_ACL_OP(input, tensor1, tensor2, value);
    op_api::addcmul_(inputAt, tensor1At, tensor2At, valueAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
