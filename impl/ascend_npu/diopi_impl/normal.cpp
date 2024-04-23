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
    if (outAt.numel() > 0) {
        op_api::normal_out(mean, std, outAt.sizes(), generatorAt, outAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(inout, generator);
    if (inoutAt.numel() > 0) {
        op_api::normal_(inoutAt, mean, std, generatorAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
