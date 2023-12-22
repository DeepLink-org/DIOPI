/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    BEGIN_CALL_ACL_OP(input, out, other);
    std::cout << "input.size()=" << inputAt.sizes() << ", other.size()=" << otherAt.sizes() << ", out.size()=" << outAt.sizes() << ", input.device()=" << inputAt.device() << ", other.device()=" << otherAt.device() << ", out.device()=" << outAt.device() << std::endl;
    acl_op::matmul_out(inputAt, otherAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
