/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" {
DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    std::vector<diopiConstTensorHandle_t> dynamicInput(tensors, tensors + numInputs);
    AclOpRunner<1, 1>("Pack", ctx).addDynamicInput(dynamicInput).setAttr("N", numInputs).setAttr("axis", dim).addOutput(out).run();
    return diopiSuccess;
}
}

}  // namespace ascend
}  // namespace impl
