/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AscendTensor outTensor(out);
    AclOpRunner<2, 1>("Expand", ctx).addInput(input).addConstInput(outTensor.shape()).addOutput(out).run();
}

}  // namespace ascend
}  // namespace impl
