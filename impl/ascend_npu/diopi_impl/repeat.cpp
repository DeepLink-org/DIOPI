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
    std::cout << std::endl;
    std::cout << "repeatPara = ";
    for (int i = 0; i < repeatSize.len; i++) {
        std::cout << repeatSize.data[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "origin inputShape = ";
    for (int64_t i : inputShape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    if (inputShape.size() < repeatSize.len) {
        while (inputShape.size() < repeatSize.len) {
            inputShape.insert(inputShape.begin(), 1);
        }

        inputAt = impl::aten::view(inputAt, inputShape);
    }
    std::cout << "modified inputShape = ";
    for (int64_t i : inputShape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    outAt = acl_op::repeat(inputAt, repeatSizeAt);

    std::cout << "outShape = ";
    std::vector<int64_t> outShape(outAt.sizes().cbegin(), outAt.sizes().cend());
    for (int64_t i : outShape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "inputTensor.dtype = " << inputAt.dtype() << std::endl;
    std::cout << std::endl;
    END_CALL_ACL_OP();
}

}  // extern C
