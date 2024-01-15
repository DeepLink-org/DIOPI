/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    AclOpRunner<2, 1>("GatherElements", ctx).addInput(input).addInput(index).setAttr("dim", dim).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                 int64_t dim, diopiConstTensorHandle_t index) {
    AscendTensor gradInputAt(gradInput);
    diopiScalar_t scalarZero = constructDiopiScalarT(gradInputAt.dtype(), 0);
    diopiFill(ctx, gradInput, &scalarZero);
    AclOpRunner<3, 1>("ScatterAddWithAxis", ctx).addInput(gradInput).addInput(index).addInput(gradOutput).setAttr("axis", dim).addOutput(gradInput).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
