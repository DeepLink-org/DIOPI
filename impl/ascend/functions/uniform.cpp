/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiGeneratorHandle_t generator) {
    const std::pair<uint64_t, uint64_t> gen = getSeedAndOffset(ctx, generator, 10);
    const uint64_t seed = gen.first;
    const uint64_t offset = gen.second;
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceUniform, ctx, inout, from, to, seed, offset);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
