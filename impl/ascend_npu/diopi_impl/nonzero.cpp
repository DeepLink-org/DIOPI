/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
namespace OP_IMPL_NS {

diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    BEGIN_CALL_ACL_OP(input);
    at::Tensor outAt = op_api::nonzero(inputAt);
    impl::aten::buildDiopiTensor(ctx, outAt, out);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
