/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                       bool descending, const bool* stable) {
    bool tem = *stable;
    if (dim == 0)
    {
        AscendTensor tensor(input);
        dim = tensor.dim()-1;
    }
    AclOpRunner<1, 2>("Sort", ctx)
        .addInput(input)
        .setAttr("axis", dim)
        .setAttr("descending", descending)
        .setAttr("stable", tem)
        .addOutput(values)
        .addOutput(indices)
        .run();

    // AscendTensor tem(input);
    // AclOpRunner<2, 2>("TopKV2", ctx)
    //     .addInput(input)
    //     .addConstInput(tem.shape())
    //     .setAttr("dim", dim)
    //     .setAttr("largest", descending)
    //     .setAttr("sorted", (bool)1)
    //     .addOutput(values)
    //     .addOutput(indices)
    //     .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
