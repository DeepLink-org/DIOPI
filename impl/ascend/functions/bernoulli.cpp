/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
static const uint64_t philoxDefaultNum = 10;

diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiGeneratorHandle_t generator) {
    const std::pair<uint64_t, uint64_t> gen = getSeedAndOffset(ctx, generator, philoxDefaultNum);
    const uint64_t seed = gen.first;
    const uint64_t offset = gen.second;
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceBernoulliTensor, ctx, out, input, seed, offset);
    return diopiSuccess;
}

diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiGeneratorHandle_t generator) {
    const std::pair<uint64_t, uint64_t> gen = getSeedAndOffset(ctx, generator, philoxDefaultNum);
    const uint64_t seed = gen.first;
    const uint64_t offset = gen.second;
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceBernoulliTensor, ctx, inout, inout, seed, offset);
    return diopiSuccess;
}

diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, diopiGeneratorHandle_t generator) {
    const std::pair<uint64_t, uint64_t> gen = getSeedAndOffset(ctx, generator, philoxDefaultNum);
    const uint64_t seed = gen.first;
    const uint64_t offset = gen.second;
    auto pScalar = constructDiopiScalarT(diopi_dtype_float64, p);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceBernoulli, ctx, out, &pScalar, seed, offset);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
