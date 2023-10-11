/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Floor", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiFloor(ctx, input, input); }

}  // namespace ascend
}  // namespace impl
