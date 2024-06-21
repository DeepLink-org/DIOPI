/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

extern "C" {
diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeatSize) {
    BEGIN_CALL_ACL_OP(out, input, repeatSize);
    TORCH_CHECK(inputAt.dim() <= repeatSize.len, "repeats size should not be smaller than input tensor dim on ascend!");
    // When repeatSize.len is equal to 0, out is the same as input.
    if (repeatSize.len == 0) {
        outAt.copy_(inputAt, true);
        END_CALL_ACL_OP();
    }

    std::vector<int64_t> inputShape = inputAt.sizes().vec();
    inputShape.insert(inputShape.begin(), repeatSize.len - inputAt.dim(), 1);
    inputAt = impl::aten::viewStorage(inputAt, inputShape);

    EXEC_NPU_CMD(aclnnRepeat, inputAt, repeatSizeAt, outAt);
    END_CALL_ACL_OP();
}

}  // extern C
