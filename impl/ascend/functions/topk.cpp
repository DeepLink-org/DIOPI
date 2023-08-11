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
DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                                 int64_t dim, bool largest, bool sorted) {
    std::vector<int64_t> kVec({k});
    diopiSize_t kSize(kVec.data(), static_cast<int64_t>(kVec.size()));
    AclOpRunner<2, 2>("TopKV2", ctx)
        .addInput(input)
        .addConstInput(kSize)
        .setAttr("dim", dim)
        .setAttr("largest", largest)
        .setAttr("sorted", sorted)
        .addOutput(values, indices)
        .run();
    return diopiSuccess;
}
}

}  // namespace ascend
}  // namespace impl
