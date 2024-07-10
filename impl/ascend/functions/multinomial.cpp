/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"
#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numSamples, bool replacement,
                              diopiGeneratorHandle_t generator) {
    std::pair<uint64_t, int64_t> pair = getSeedAndOffset(ctx, generator, 10);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMultinomial, ctx, input, numSamples, replacement, static_cast<int64_t>(pair.first), pair.second, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
