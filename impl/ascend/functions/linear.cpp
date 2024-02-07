/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                         diopiConstTensorHandle_t bias) {
    // convert inputs to AscendTensor
    AscendTensor inputAt(input);
    AscendTensor outputAt(out);
    AscendTensor weightAt(weight);

    if (inputAt.numel() == 0 || weightAt.numel() == 0) {
        diopiScalar_t zero = constructDiopiScalarT(outputAt.dtype(), 0.0);
        diopiFill(ctx, out, &zero);
        return diopiSuccess;
    }

    AclOpRunner<3, 1> runner("MatMulV2", ctx);
    runner.addInput(inputAt).addInput(weightAt).setAttr<uint8_t>("transpose_x1", false).setAttr<uint8_t>("transpose_x2", true).addOutput(outputAt).run();

    // if bias is not nullptr, add bias to input
    if (bias) {
        diopiScalar_t oneScalar = constructDiopiScalarT(outputAt.dtype(), 1.0);
        diopiAddInp(ctx, out, bias, &oneScalar);
    }

    return diopiSuccess;
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                 diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    AscendTensor gradWeightAt(gradWeight);
    AscendTensor gradOutputAt(gradOutput);
    AscendTensor inputAt(input);
    AscendTensor weightAt(weight);

    if (gradOutputAt.numel() == 0 || weightAt.numel() == 0 || inputAt.numel() == 0) {
        diopiScalar_t zero = constructDiopiScalarT(inputAt.dtype(), 0.0);
        diopiFill(ctx, gradInput, &zero);
        diopiFill(ctx, gradWeight, &zero);
        diopiFill(ctx, gradBias, &zero);
        return diopiSuccess;
    }

    if (nullptr != gradInput) {
        AscendTensor gradInputAt(gradInput);
        AclOpRunner<2, 1>("MatMul", ctx)
            .addInput(gradOutputAt)
            .addInput(weightAt)
            .setAttr<uint8_t>("transpose_x1", false)
            .setAttr<uint8_t>("transpose_x2", false)
            .addOutput(gradInputAt)
            .run();
    }

    if (inputAt.shape().size() > 2) transTensorTo2D(ctx, inputAt);

    if (nullptr != gradWeight) {
        if (gradWeightAt.shape().size() > 2) transTensorTo2D(ctx, gradWeightAt);

        AclOpRunner<2, 1>("MatMul", ctx)
            .addInput(gradOutputAt)
            .addInput(inputAt)
            .setAttr<uint8_t>("transpose_x1", true)
            .setAttr<uint8_t>("transpose_x2", false)
            .addOutput(gradWeightAt)
            .run();
    }

    AscendTensor reshapedGradOutputAt;
    makeTensorLike(ctx, reshapedGradOutputAt, gradOutputAt, gradOutputAt.dtype());
    reshape(ctx, gradOutputAt, reshapedGradOutputAt, gradOutputAt.shape());

    diopiTensorHandle_t diopiGradOutputAt = const_cast<diopiTensorHandle_t>(reshapedGradOutputAt.tensorHandle());
    if (gradBias) {
        std::vector<int64_t> dimVec(gradOutputAt.shape().size() - 1);
        std::iota(std::begin(dimVec), std::end(dimVec), 0);
        diopiSize_t dim = vectorToDiopiSize(dimVec);
        diopiSum(ctx, gradBias, diopiGradOutputAt, dim);
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
