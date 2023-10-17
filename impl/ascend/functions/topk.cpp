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
    diopiDtype_t outDtype, castType;
    diopiGetTensorDtype(values, &outDtype);

    if (isFloatingType(outDtype)) {
        castType = diopi_dtype_float32;
    } else {
        castType = diopi_dtype_int32;
    }

    AscendTensor outA(values);
    castTensor(ctx, outA, castType);

    AclOpRunner<2, 2>("TopKV2", ctx)
        .addInput(input, castType)
        .addConstInput(kSize)
        .setAttr("dim", dim)
        .setAttr("largest", largest)
        .setAttr("sorted", sorted)
        .addOutput(outA)
        .addOutput(indices)
        .run();

    diopiCastDtype(ctx, values, static_cast<diopiConstTensorHandle_t>(outA));

    AscendTensor o(indices);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
