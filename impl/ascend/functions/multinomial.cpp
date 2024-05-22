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
    auto pair = getSeedAndOffset(ctx, generator, 10);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMultinomial, ctx, input, numSamples, replacement, pair.first, pair.second, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
