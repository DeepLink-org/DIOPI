/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    AclOpRunner<2, 1>("ReverseV2", ctx).addInput(input).addConstInput(dims).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
