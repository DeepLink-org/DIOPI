/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
DIOPI_API diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numSamples,
                                        bool replacement) {
    AclOpRunner<3, 1>("MultinomialWithReplacement", ctx)
        .addInput(input)
        .addConstInput(0, diopi_dtype_int64)
        .addConstInput(0, diopi_dtype_int64)
        .setAttr("numsamples", numSamples)
        .setAttr("replacement", replacement)
        .addOutput(out)
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
