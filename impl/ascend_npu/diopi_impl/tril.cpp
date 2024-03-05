/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    BEGIN_CALL_ACL_OP(input, out);
    op_api::tril_out(inputAt, diagonal, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
