/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    BEGIN_CALL_ACL_OP(out, start, end);
    op_api::linspace_out(startAt, endAt, steps, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
