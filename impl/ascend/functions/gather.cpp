/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    AclOpRunner<2, 1>("GatherElements", ctx).addInput(input).addInput(index).setAttr("dim", dim).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
