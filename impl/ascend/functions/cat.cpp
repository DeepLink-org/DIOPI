/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    std::vector<diopiConstTensorHandle_t> dynamicInput(tensors, tensors + numInputs);
    AclOpRunner<1, 1>("ConcatD", ctx).addDynamicInput(dynamicInput).setAttr("N", numInputs).setAttr("concat_dim", dim).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
