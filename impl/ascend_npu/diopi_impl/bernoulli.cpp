/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <c10/core/Scalar.h>

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator);
    auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(generatorAt)->philox_engine_inputs(10);
    const uint64_t seed = pair.first;
    const uint64_t offset = pair.second;
    auto pScalar = at::Scalar(p);
    EXEC_NPU_CMD(aclnnInplaceBernoulli, outAt, pScalar, seed, offset);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
