/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    AscendTensor inputAt(input);
    AscendTensor mat2At(mat2);
    AscendTensor outputAt(out);
    if (inputAt.numel() == 0 || mat2At.numel() == 0) {
        diopiScalar_t zero = constructDiopiScalarT(outputAt.dtype(), 0.0);
        diopiFill(ctx, out, &zero);
        return diopiSuccess;
    }

    AclOpRunner<2, 1>("BatchMatMulV2", ctx).addInput(input).addInput(mat2).setAttr("adj_x1", false).setAttr("adj_x1", false).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
