/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                       int64_t dim, bool largest, bool sorted) {
    std::vector<int64_t> kVec({k});
    diopiSize_t kSize = vectorToDiopiSize(kVec);
    AclOpRunner<2, 2>("TopKV2", ctx)
        .addInput(input)
        .addConstInput(kSize)
        .setAttr("dim", dim)
        .setAttr("largest", largest)
        .setAttr("sorted", sorted)
        .addOutput(values)
        .addOutput(indices)
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
