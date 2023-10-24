/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    AclOpRunner<1, 1>("Tril", ctx).addInput(input).setAttr("diagonal", diagonal).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
