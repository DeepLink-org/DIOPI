/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    AscendTensor inputTr(input);
    AscendTensor mat2Tr(mat2);
    AscendTensor outputTr(out);
    if (inputTr.numel() == 0 || mat2Tr.numel() == 0) {
        diopiScalar_t zero = constructDiopiScalarT(outputTr.dtype(), 0.0);
        diopiFill(ctx, out, &zero);
        return diopiSuccess;
    }
    AclOpRunner<2, 1>("MatMul", ctx)
        .addInput(input, inputTr.dtype())
        .addInput(mat2, mat2.dtype())
        .setAttr("adj_x1", false)
        .setAttr("adj_x1", false)
        .addOutput(out)
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
