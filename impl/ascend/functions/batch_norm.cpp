/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean,
                                                 diopiTensorHandle_t saveInvstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                                 diopiConstTensorHandle_t bias, diopiTensorHandle_t runningMean, diopiTensorHandle_t runningVar, bool training,
                                                 double momentum, double eps) {
    if (!training) {
        AclOpRunner<5, 1>("BNInfer", ctx)
            .addInput(input, weight, bias, runningMean, runningVar)
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
        AclOpRunner<1, 2>("BNTrainingReduce", ctx).addInput(input).setAttr("epsilon", static_cast<float>(eps)).addOutput(sum, squareSum).run();
        AclOpRunner<7, 5>("BNTrainingUpdate", ctx)
            .addInput(input, sum, squareSum, weight, bias, runningMean, runningVar)
            .setAttr("epsilon", static_cast<float>(eps))
            .setAttr("factor", static_cast<float>(momentum))
            .addOutput(out, runningMean, runningMean, saveMean, saveInvstd)
            .run();
    }
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                                         diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                                         diopiConstTensorHandle_t weight, diopiConstTensorHandle_t runningMean,
                                                         diopiConstTensorHandle_t runningVar, diopiConstTensorHandle_t saveMean,
                                                         diopiConstTensorHandle_t saveInvstd, bool training, double eps) {
    if (!training) {
        AclOpRunner<3, 1>("BNInferGrad", ctx).addInput(gradOutput, weight, runningVar).setAttr("epsilon", static_cast<float>(eps)).addOutput(gradInput).run();
    } else {
        AclOpRunner<4, 2>("BNTrainingUpdateGrad", ctx)
            .addInput(gradOutput, input, saveMean, saveInvstd)
            .setAttr("epsilon", static_cast<float>(eps))
            .addOutput(gradWeight, gradBias)
            .run();
        AclOpRunner<7, 1>("BNTrainingReduceGrad", ctx)
            .addInput(gradOutput, input, gradWeight, gradBias, weight, saveMean, saveInvstd)
            .setAttr("epsilon", static_cast<float>(eps))
            .addOutput(gradInput)
            .run();
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
