/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    BEGIN_CALL_ACL_OP(out, input);
    int64_t dimTmp;
    if (dim == nullptr) {
        dimTmp = 0;
        std::vector<int64_t> flattenShape{inputAt.numel()};
        auto flattenInput = inputAt.view(flattenShape);
        EXEC_NPU_CMD(aclnnArgMax, flattenInput, dimTmp, keepdim, outAt);
    } else {
        dimTmp = *dim;
        EXEC_NPU_CMD(aclnnArgMax, inputAt, dimTmp, keepdim, outAt);
    }
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
