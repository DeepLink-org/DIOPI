/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numSamples, bool replacement,
                              diopiGeneratorHandle_t generator) {
    auto pair = getSeedAndOffset(ctx, generator, 10);
    AclOpRunner<3, 1>("MultinomialWithReplacement", ctx)
        .addInput(input)
        .addConstInput(pair.first, diopi_dtype_int64)
        .addConstInput(pair.second, diopi_dtype_int64)
        .setAttr("numsamples", numSamples)
        .setAttr("replacement", replacement)
        .addOutput(out)
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
