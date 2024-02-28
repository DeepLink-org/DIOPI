/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator);
    op_api::normal_out(mean, std, outAt.sizes(), c10::make_optional(std::move(generatorAt)), outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    return OP_IMPL_NS::diopiNormal(ctx, inout, mean, std, generator);
}

}  // namespace OP_IMPL_NS
