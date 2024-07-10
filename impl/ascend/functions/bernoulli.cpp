/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
static const uint64_t PHILOX_DEFAULT_NUM = 10;

diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, diopiGeneratorHandle_t generator) {
    const std::pair<uint64_t, uint64_t> gen = getSeedAndOffset(ctx, generator, 10);
    const uint64_t seed = gen.first;
    const uint64_t offset = gen.second;
    auto pScalar = constructDiopiScalarT(diopi_dtype_float64, p);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceBernoulli, ctx, out, p, seed, offset);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
