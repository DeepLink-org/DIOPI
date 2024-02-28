/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    BEGIN_CALL_ACL_OP(input, value);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::fill_(inputAt, valueAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
