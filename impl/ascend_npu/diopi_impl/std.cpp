/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    BEGIN_CALL_ACL_OP(out, input, dim);
    bool keepdim = false;
    if (inputAt.dim() == outAt.dim()) {
        keepdim = true;
    }
    int64_t correction = static_cast<int64_t>(unbiased);
    EXEC_NPU_CMD(aclnnStd, inputAt, dimAt, correction, keepdim, outAt);
    return diopiSuccess;
}

}  // namespace OP_IMPL_NS
