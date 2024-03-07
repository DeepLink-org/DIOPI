/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiAtan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    op_api::atan_out(inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiAtanInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    op_api::atan_(inputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
