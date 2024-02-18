/**
 * @file
 * @author DeepLink
 * @Atright  (c) 2023, DeepLink.
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

    // mm's input matrix must be 2D, it needs to be converted if it isn't
    // the shape of input is [*, inputFeatures]
    // the shape of weight is [outFeatures, inputFeatures]
    // the shape of output is [*, outputFeatures]
    if (inputAt.shape().size() > 2) {
        transTensorTo2D(ctx, inputAt);
    }
    if (outputAt.shape().size() > 2) {
        transTensorTo2D(ctx, outputAt);
    }

    ASCEND_CHECK_ABORT(inputAt.shape()[inputAt.shape().size() - 1] == weightAt.shape()[1], "the last dim of weight must be inFeatures");
    ASCEND_CHECK_ABORT(outputAt.shape()[outputAt.shape().size() - 1] == weightAt.shape()[0], "the first dim of weight must be outFeatures");

    AclOpRunner<3, 1> runner("MatMulV2", ctx);
    runner.addInput(inputAt).addInput(weightAt).setAttr<uint8_t>("transpose_x1", false).setAttr<uint8_t>("transpose_x2", true).addOutput(outputAt).run();

    // if bias is not nullptr, also add bias to input
    if (bias) {
        diopiScalar_t oneScalar = constructDiopiScalarT(outputAt.dtype(), 1.0);
        diopiAdd(ctx, out, out, bias, &oneScalar);
    }
    return diopiSuccess;
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                 diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    AscendTensor gradWeightAt(gradWeight);
    AscendTensor gradOutputAt(gradOutput);
    AscendTensor inputAt(input);
    AscendTensor weightAt(weight);

    const std::vector<int64_t> gradInputPrimaryShape = inputAt.shape();
    bool transTensorTo2DFalg = false;

    if (gradOutputAt.numel() == 0 || weightAt.numel() == 0 || inputAt.numel() == 0) {
        diopiScalar_t zero = constructDiopiScalarT(inputAt.dtype(), 0.0);
        diopiFill(ctx, gradInput, &zero);
        diopiFill(ctx, gradWeight, &zero);
        diopiFill(ctx, gradBias, &zero);
        return diopiSuccess;
    }

    if (weightAt.shape().size() > 2) transTensorTo2D(ctx, weightAt);
    if (gradOutputAt.shape().size() > 2) transTensorTo2D(ctx, gradOutputAt);

    if (nullptr != gradInput) {
        AscendTensor gradInputAt(gradInput);
        if (inputAt.shape().size() > 2) {
            transTensorTo2DFalg = true;
            transTensorTo2D(ctx, gradInputAt);
        }

        AclOpRunner<2, 1>("MatMul", ctx)
            .addInput(gradOutputAt)
            .addInput(weightAt)
            .setAttr<uint8_t>("transpose_x1", false)
            .setAttr<uint8_t>("transpose_x2", false)
            .addOutput(gradInputAt)
            .run();

        if (transTensorTo2DFalg) {
            gradInputAt.view(gradInputPrimaryShape);
        }
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
