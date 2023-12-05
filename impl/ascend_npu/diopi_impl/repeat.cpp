/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

extern "C" {
diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeatSize) {
    BEGIN_CALL_ACL_OP(out, input, repeatSize);
    std::vector<int64_t> inputShape(inputAt.sizes().cbegin(), inputAt.sizes().cend());

    if (inputShape.size() < repeatSize.len) {
        while (inputShape.size() < repeatSize.len) {
            inputShape.insert(inputShape.begin(), 1);
        }

        inputAt = impl::aten::view(inputAt, inputShape);
    }

    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    outAt = acl_op::repeat(inputAt, repeatSizeAt);
    END_CALL_ACL_OP();
}

}  // extern C
