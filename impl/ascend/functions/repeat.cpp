/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeatsSize) {
    AclOpRunner<2, 1>("Tile", ctx).addInput(input).addConstInput(repeatsSize).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
