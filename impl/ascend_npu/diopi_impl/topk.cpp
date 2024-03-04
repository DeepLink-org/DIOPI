/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                       int64_t dim, bool largest, bool sorted) {
    BEGIN_CALL_ACL_OP(values, indices, input);
    op_api::topk_out(inputAt, k, dim, largest, sorted, valuesAt, indicesAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
