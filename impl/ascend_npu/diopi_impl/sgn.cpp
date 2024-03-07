/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiSgn(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input, out);
    EXEC_NPU_CMD(aclnnSign, inputAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiSgnInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    EXEC_NPU_CMD(aclnnSign, inputAt, inputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
