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
    // std::cout << "--------------------------------------------------------------------------------------------------------" << std::endl;
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

    AscendTensor tem(input);
    if (tem.dim() == 0 && k == 1) {
        // std::cout << "innnnnnnnnnnnnnnnnnnnnnnn" << std::endl;
        AscendTensor outB(input);
        castTensor(ctx, outB, outDtype);
        diopiCastDtype(ctx, values, static_cast<diopiConstTensorHandle_t>(outB));

        diopiScalar_t zero = constructDiopiScalarT(diopi_dtype_int32, 0);
        diopiFill(ctx, indices, &zero);
        return diopiSuccess;
    }

    AclOpRunner<2, 2>("TopKV2", ctx)
        .addInput(input, castType)
        .addConstInput(kSize)
        .setAttr("dim", dim)
        .setAttr("largest", largest)
        .setAttr("sorted", sorted)
        .addOutput(outA)
        .addOutput(indices)
        .run();

    castTensor(ctx, outA, outDtype);
    diopiCastDtype(ctx, values, static_cast<diopiConstTensorHandle_t>(outA));

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
