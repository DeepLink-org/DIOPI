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
    AscendTensor gradWeightCopy(gradWeight);
    AscendTensor gradOutputCopy(gradOutput);
    AscendTensor inputCopy(input);
    AscendTensor weightCopy(weight);

    const std::vector<int64_t> gradInputPrimaryShape = inputCopy.shape();
    bool transTensorTo2DFalg = false;

    if (gradOutputCopy.numel() == 0 || weightCopy.numel() == 0 || inputCopy.numel() == 0) {
        diopiScalar_t zero = constructDiopiScalarT(inputCopy.dtype(), 0.0);
        diopiFill(ctx, gradInput, &zero);
        diopiFill(ctx, gradWeight, &zero);
        diopiFill(ctx, gradBias, &zero);
        return diopiSuccess;
    }

    if (weightCopy.shape().size() > 2) transTensorTo2D(ctx, weightCopy);
    if (gradOutputCopy.shape().size() > 2) transTensorTo2D(ctx, gradOutputCopy);

    if (nullptr != gradInput) {
        AscendTensor gradInputCopy(gradInput);
        if (inputCopy.shape().size() > 2) {
            transTensorTo2DFalg = true;
            transTensorTo2D(ctx, gradInputCopy);
        }

        AclOpRunner<2, 1>("MatMul", ctx)
            .addInput(gradOutputCopy)
            .addInput(weightCopy)
            .setAttr<uint8_t>("transpose_x1", false)
            .setAttr<uint8_t>("transpose_x2", false)
            .addOutput(gradInputCopy)
            .run();

        if (transTensorTo2DFalg) {
            gradInputCopy.view(gradInputPrimaryShape);
        }
    }

    if (inputCopy.shape().size() > 2) transTensorTo2D(ctx, inputCopy);

    if (nullptr != gradWeight) {
        if (gradWeightCopy.shape().size() > 2) transTensorTo2D(ctx, gradWeightCopy);

        AclOpRunner<2, 1>("MatMul", ctx)
            .addInput(gradOutputCopy)
            .addInput(inputCopy)
            .setAttr<uint8_t>("transpose_x1", true)
            .setAttr<uint8_t>("transpose_x2", false)
            .addOutput(gradWeightCopy)
            .run();
    }

    AscendTensor reshapedGradOutputCopy;
    makeTensorLike(ctx, reshapedGradOutputCopy, gradOutputCopy, gradOutputCopy.dtype());
    reshape(ctx, gradOutputCopy, reshapedGradOutputCopy, gradOutputCopy.shape());

    diopiTensorHandle_t diopiGradOutputCopy = const_cast<diopiTensorHandle_t>(reshapedGradOutputCopy.tensorHandle());
    if (gradBias) {
        std::vector<int64_t> dimVec(gradOutputCopy.shape().size() - 1);
        std::iota(std::begin(dimVec), std::end(dimVec), 0);
        diopiSize_t dim = vectorToDiopiSize(dimVec);
        diopiSum(ctx, gradBias, diopiGradOutputCopy, dim);
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
