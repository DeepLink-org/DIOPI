
/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    BEGIN_CALL_ACL_OP(start, end, step, out);
    op_api::arange_out(startAt, endAt, stepAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
