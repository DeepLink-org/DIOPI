/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numSamples, bool replacement,
                              diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(input, generator, out);
    op_api::multinomial_out(inputAt, numSamples, replacement, generatorAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
