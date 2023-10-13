/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

void updateInputAscendTensorDim(AscendTensor &inputAt, bool training) {
    int64_t dim = inputAt.dim();
    if (2 == dim) {
        inputAt.unsqueeze(2);
        inputAt.unsqueeze(3);
    } else if (3 == dim) {
        inputAt.unsqueeze(3);
    } else if (5 == dim && !training) {
        std::vector<int64_t> shape4d{inputAt.shape(0), inputAt.shape(1), inputAt.shape(2), inputAt.shape(3) * inputAt.shape(4)};
        inputAt.view(shape4d);
    }
}

void batchNormBackwardTrainingUpdate(diopiContextHandle_t ctx, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias, AscendTensor gradOutputAt,
                                     AscendTensor inputAt, diopiConstTensorHandle_t saveMean, diopiConstTensorHandle_t saveInvstd, double eps) {
    std::string name = (inputAt.dim() == 5) ? "BN3DTrainingUpdateGrad" : "BNTrainingUpdateGrad";
    AclOpRunner<4, 2>(name, ctx)
        .addInput(gradOutputAt)
        .addInput(inputAt)
        .addInput(saveMean)
        .addInput(saveInvstd)
        .addOutput(gradWeight)
        .addOutput(gradBias)
        .setAttr<float>("epsilon", static_cast<float>(eps))
        .run();
}

void batchNormBackwardTrainingReduceNocheck(diopiContextHandle_t ctx, AscendTensor gradInputAt, diopiConstTensorHandle_t gradWeight,
                                            diopiConstTensorHandle_t gradBias, AscendTensor gradOutputAt, AscendTensor inputAt, diopiConstTensorHandle_t weight,
                                            diopiConstTensorHandle_t saveMean, diopiConstTensorHandle_t saveInvstd, double eps) {
    std::string name = (inputAt.dim() == 5) ? "BN3DTrainingReduceGrad" : "BNTrainingReduceGrad";
    AclOpRunner<7, 1>(name, ctx)
        .addInput(gradOutputAt)
        .addInput(inputAt)
        .addInput(gradWeight)
        .addInput(gradBias)
        .addInput(weight)
        .addInput(saveMean)
        .addInput(saveInvstd)
        .addOutput(gradInputAt)
        .setAttr<float>("epsilon", static_cast<float>(eps))
        .run();
}

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t runningMean,
                            diopiTensorHandle_t runningVar, bool training, double momentum, double eps) {
    if (runningMean == nullptr) {
        makeTensorLike(ctx, &runningMean, weight, diopi_dtype_float32);
        diopiScalar_t zero = {diopi_dtype_float32, {0}};
        diopiFill(ctx, runningMean, &zero);
    }
    if (runningVar == nullptr) {
        makeTensorLike(ctx, &runningVar, weight, diopi_dtype_float32);
        diopiScalar_t one = {diopi_dtype_float32, {1}};
        diopiFill(ctx, runningMean, &one);
    }

    AscendTensor inputAt(input), outputAt(out);
    updateInputAscendTensorDim(inputAt, training);
    outputAt.view(inputAt.getAclMemShape());

    if (!training) {
        AclOpRunner<5, 1>("BNInfer", ctx)
            .addInput(inputAt)
            .addInput(weight)
            .addInput(bias)
            .addInput(runningMean)
            .addInput(runningVar)
            .addOutput(outputAt)
            .setAttr("epsilon", static_cast<float>(eps))
            .run();

        diopiTensorHandle_t runningVarBroadcasted;
        makeTensorLike(ctx, &runningVarBroadcasted, input);
        AscendTensor runningVarAt(runningVar);
        runningVarAt.unsqueeze(0);
        runningVarAt.unsqueeze(2);
        runningVarAt.unsqueeze(3);
        AclOpRunner<2, 1>("BroadcastTo", ctx).addInput(runningVarAt).addConstInput(inputAt.shape()).addOutput(runningVarBroadcasted).run();
        negativeInputRtnFillNan(ctx, out, runningVarBroadcasted);
    } else {
        diopiTensorHandle_t sum = nullptr, squareSum = nullptr;
        diopiSize_t shape, stride;
        diopiGetTensorShape(runningMean, &shape);
        diopiGetTensorStride(runningMean, &stride);
        diopiRequireTensor(ctx, &sum, &shape, &stride, diopiDtype_t::diopi_dtype_float32, diopi_device);
        diopiRequireTensor(ctx, &squareSum, &shape, &stride, diopiDtype_t::diopi_dtype_float32, diopi_device);
        AclOpRunner<1, 2>("BNTrainingReduce", ctx).addInput(inputAt).setAttr("epsilon", static_cast<float>(eps)).addOutput(sum).addOutput(squareSum).run();
        AclOpRunner<7, 5>("BNTrainingUpdate", ctx)
            .addInput(inputAt)
            .addInput(sum)
            .addInput(squareSum)
            .addInput(weight)
            .addInput(bias)
            .addInput(runningMean)
            .addInput(runningVar)
            .setAttr("epsilon", static_cast<float>(eps))
            .setAttr("factor", static_cast<float>(momentum))
            .addOutput(outputAt)
            .addOutput(runningMean)
            .addOutput(runningVar)
            .addOutput(saveMean)
            .addOutput(saveInvstd)
            .run();
    }
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t runninMean, diopiConstTensorHandle_t runningVar, diopiConstTensorHandle_t saveMean,
                                    diopiConstTensorHandle_t saveInvstd, bool training, double eps) {
    AscendTensor inputAt(input), gradOutputAt(gradOutput), gradInputAt(gradInput);
    updateInputAscendTensorDim(inputAt, training);
    gradOutputAt.view(inputAt.getAclMemShape());
    gradInputAt.view(inputAt.getAclMemShape());

    if (!training) {
        batchNormBackwardTrainingUpdate(ctx, gradWeight, gradBias, gradOutputAt, inputAt, runninMean, runningVar, eps);
        negativeInputRtnFillNan(ctx, gradWeight, runningVar);

        AclOpRunner<3, 1>("BNInferGrad", ctx)
            .addInput(gradOutputAt)
            .addInput(weight)
            .addInput(runningVar)
            .addOutput(gradInputAt)
            .setAttr<float>("epsilon", static_cast<float>(eps))
            .run();

        diopiTensorHandle_t runningVarBroadcasted;
        makeTensorLike(ctx, &runningVarBroadcasted, input);
        AscendTensor runningVarAt(runningVar);
        runningVarAt.unsqueeze(0);
        runningVarAt.unsqueeze(2);
        runningVarAt.unsqueeze(3);
        AclOpRunner<2, 1>("BroadcastTo", ctx).addInput(runningVarAt).addConstInput(inputAt.shape()).addOutput(runningVarBroadcasted).run();
        negativeInputRtnFillNan(ctx, gradInput, runningVarBroadcasted);
    } else {
        batchNormBackwardTrainingUpdate(ctx, gradWeight, gradBias, gradOutputAt, inputAt, saveMean, saveInvstd, eps);
        batchNormBackwardTrainingReduceNocheck(ctx, gradInputAt, gradWeight, gradBias, gradOutputAt, inputAt, weight, saveMean, saveInvstd, eps);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
