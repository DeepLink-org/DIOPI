/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

void batchNormBackwardTrainingUpdate(diopiContextHandle_t ctx, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOut,
                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t saveMean, diopiConstTensorHandle_t saveInvstd, double eps) {
    diopiSize_t inputSize;
    diopiGetTensorShape(input, &inputSize);
    std::string name = (inputSize.len == 5) ? "BN3DTrainingUpdateGrad" : "BNTrainingUpdateGrad";
    auto format = (inputSize.len == 5) ? ACL_FORMAT_NCDHW : ACL_FORMAT_NCHW;

    AclOpRunner<4, 2>(name, ctx)
        .addInput(gradOut, format)
        .addInput(input, format)
        .addInput(saveMean, format)
        .addInput(saveInvstd, format)
        .addOutput(gradWeight, format)
        .addOutput(gradBias, format)
        .setAttr<float>("epsilon", static_cast<float>(eps))
        .run();
}

void batchNormBackwardTrainingReduceNocheck(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradWeight,
                                            diopiConstTensorHandle_t gradBias, diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t saveMean, diopiConstTensorHandle_t saveInvstd,
                                            double eps) {
    diopiSize_t inputSize;
    diopiGetTensorShape(input, &inputSize);
    std::string name = (inputSize.len == 5) ? "BN3DTrainingReduceGrad" : "BNTrainingReduceGrad";
    auto format = (inputSize.len == 5) ? ACL_FORMAT_NCDHW : ACL_FORMAT_NCHW;
    AclOpRunner<7, 1>(name, ctx)
        .addInput(gradOut, format)
        .addInput(input, format)
        .addInput(gradWeight, format)
        .addInput(gradBias, format)
        .addInput(weight, format)
        .addInput(saveMean, format)
        .addInput(saveInvstd, format)
        .addOutput(gradInput, format)
        .setAttr<float>("epsilon", static_cast<float>(eps))
        .run();
}

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t runningMean,
                            diopiTensorHandle_t runningVar, bool training, double momentum, double eps) {
    if (!training) {
        AclOpRunner<5, 1>("BNInfer", ctx)
            .addInput(input)
            .addInput(weight)
            .addInput(bias)
            .addInput(runningMean)
            .addInput(runningVar)
            .addOutput(out)
            .setAttr("epsilon", static_cast<float>(eps))
            .run();
    } else {
        diopiTensorHandle_t sum = nullptr, squareSum = nullptr;
        diopiSize_t shape, stride;
        diopiGetTensorShape(runningMean, &shape);
        diopiGetTensorStride(runningMean, &stride);
        diopiRequireTensor(ctx, &sum, &shape, &stride, diopiDtype_t::diopi_dtype_float32, diopi_device);
        diopiRequireTensor(ctx, &squareSum, &shape, &stride, diopiDtype_t::diopi_dtype_float32, diopi_device);
        AclOpRunner<1, 2>("BNTrainingReduce", ctx).addInput(input).setAttr("epsilon", static_cast<float>(eps)).addOutput(sum).addOutput(squareSum).run();
        AclOpRunner<7, 5>("BNTrainingUpdate", ctx)
            .addInput(input)
            .addInput(sum)
            .addInput(squareSum)
            .addInput(weight)
            .addInput(bias)
            .addInput(runningMean)
            .addInput(runningVar)
            .setAttr("epsilon", static_cast<float>(eps))
            .setAttr("factor", static_cast<float>(momentum))
            .addOutput(out)
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
    if (!training) {
        batchNormBackwardTrainingUpdate(ctx, gradWeight, gradBias, gradOutput, input, runninMean, runningVar, eps);
        AclOpRunner<3, 1>("BNInferGrad", ctx)
            .addInput(gradOutput)
            .addInput(weight)
            .addInput(runningVar)
            .addOutput(gradInput)
            .setAttr<float>("epsilon", static_cast<float>(eps))
            .run();
    } else {
        batchNormBackwardTrainingUpdate(ctx, gradWeight, gradBias, gradOutput, input, saveMean, saveInvstd, eps);
        batchNormBackwardTrainingReduceNocheck(ctx, gradInput, gradWeight, gradBias, gradOutput, input, weight, saveMean, saveInvstd, eps);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
