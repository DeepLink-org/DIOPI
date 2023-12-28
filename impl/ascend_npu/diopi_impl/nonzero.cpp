/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    at::Tensor outAt = acl_op::nonzero(inputAt);
    auto shapeAt = outAt.sizes();
    diopiSize_t outputSize;
    outputSize.data = shapeAt.data();
    outputSize.len = shapeAt.size();
    diopiTensorHandle_t output;
    diopiRequireTensor(ctx, &output, &outputSize, nullptr, diopi_dtype_int64, diopi_device);
    auto outputAt = impl::aten::buildATen(output);
    outputAt.copy_(outAt);
    *out = output;
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
