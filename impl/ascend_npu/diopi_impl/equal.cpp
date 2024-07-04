/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {
diopiError_t diopiEqual(diopiContextHandle_t ctx, bool* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, other);
    at::Tensor outAt = at_npu::native::empty_npu({1}, inputAt.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnEqual, inputAt, otherAt, outAt);
    // TODO(chenchiyu): using outAt.item<bool>() directly would cause ACL stream synchronize failed on a+k platform.
    // We should find a better way to remove the dispatch cost of .cpu().
    *out = outAt.cpu().item<bool>();
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
