/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    std::vector<int64_t> dimVec({dim});
    diopiSize_t dimInput = vectorToDiopiSize(dimVec);
    AclOpRunner<3, 1>("GatherV2", ctx).addInput(input).addInput(index).addConstInput(dimInput).setAttr<int64_t>("batch_dims", 0).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t grad, diopiSize_t inputSizes,
                                      int64_t dim, diopiConstTensorHandle_t index) {
    AscendTensor gradInputAt(gradInput);
    if (dim < 0) {
        dim = dim + inputSizes.len;
    }
    std::vector<int64_t> dimVec({dim});
    diopiSize_t dimInput = vectorToDiopiSize(dimVec);
    diopiScalar_t scalarZero = constructDiopiScalarT(gradInputAt.dtype(), 0);
    diopiFill(ctx, gradInput, &scalarZero);
    AclOpRunner<3, 1>("InplaceIndexAdd", ctx).addInput(gradInput).addInput(index).addInput(grad).setAttr<int64_t>("axis", dim).addOutput(gradInput).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
