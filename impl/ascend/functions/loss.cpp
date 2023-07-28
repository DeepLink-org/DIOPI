/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" {

DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    auto totalWeightScalar = diopiScalar_t();
    totalWeightScalar.stype = diopi_dtype_float64;
    totalWeightScalar.fval = 0.0;
    diopiTensorHandle_t totalWeight;
    makeTensorFromScalar(ctx, &totalWeightScalar, &totalWeight, diopi_dtype_float32, diopi_device);

    AclOpRunner<3, 2> runner("NLLLoss");
    runner.addInput(input, target).setAttr("ignore_index", ignoreIndex).addOutput(out, totalWeight);
    if (weight) {
        runner.addInput(weight);
    }
    if (reduction == diopiReduction_t::ReductionMean) {
        runner.setAttr("reduction", std::string("mean"));
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner.setAttr("reduction", std::string("sum"));
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner.setAttr("reduction", std::string("none"));
    }
    runner.run(ctx);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignoreIndex) {
    AclOpRunner<5, 1> runner("NLLLossGrad");
    runner.addInput(input, gradOutput, target).setAttr("ignore_index", ignoreIndex).addOutput(gradInput);
    diopiTensorHandle_t totalWeight;
    auto totalWeightScalar = diopiScalar_t();
    totalWeightScalar.stype = diopi_dtype_float64;
    totalWeightScalar.fval = 0.0;
    makeTensorFromScalar(ctx, &totalWeightScalar, &totalWeight, diopi_dtype_float32, diopi_device);
    if (weight) {
        diopiSum(ctx, totalWeight, weight, diopiSize_t());
        runner.addInput(weight, totalWeight);
    } else {
        diopiTensorHandle_t weightNew;
        diopiSize_t inputShape;
        diopiGetTensorShape(input, &inputShape);
        int64_t weightDim[] = {inputShape.data[1]};
        diopiSize_t weightShape(weightDim, 1);
        diopiRequireTensor(ctx, &weightNew, &weightShape, nullptr, diopi_dtype_float32, diopi_device);
        AclOpRunner<1, 1>("Fills").addInput(weightNew).setAttr<float>("value", 1.0).addOutput(weightNew).run(ctx);
        diopiSum(ctx, totalWeight, weightNew, diopiSize_t());
        runner.addInput(weightNew, totalWeight);
    }
    if (reduction == diopiReduction_t::ReductionMean) {
        runner.setAttr("reduction", std::string("mean"));
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner.setAttr("reduction", std::string("sum"));
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner.setAttr("reduction", std::string("none"));
    }
    runner.run(ctx);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                             diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    diopiTensorHandle_t logTensor;
    makeTensorLike(ctx, &logTensor, input);
    diopiLogSoftmax(ctx, logTensor, input, 1);
    diopiNLLLoss(ctx, out, logTensor, target, weight, reduction, ignoreIndex);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    diopiTensorHandle_t logTensor, gradLog;
    makeTensorLike(ctx, &logTensor, input);
    diopiLogSoftmax(ctx, logTensor, input, 1);
    makeTensorLike(ctx, &gradLog, gradInput);
    diopiNLLLossBackward(ctx, gradLog, gradOutput, input, target, weight, reduction, ignoreIndex);
    diopiLogSoftmaxBackward(ctx, gradInput, gradLog, logTensor, 1);
    return diopiSuccess;
}
}
}  // namespace ascend
}  // namespace impl
