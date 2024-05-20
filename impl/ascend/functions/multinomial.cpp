/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numSamples, bool replacement,
                              diopiGeneratorHandle_t generator) {
    uint64_t seed = 0;
    uint64_t offset = 0;
    diopiGeneratorGetSeedAndOffset(generator, &seed, &offset);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMultinomial, ctx, input, numSamples, replacement, seed, offset, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
