/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    auto totalWeightScalar = diopiScalar_t();
    totalWeightScalar.stype = diopi_dtype_float64;
    totalWeightScalar.fval = 0.0;
    diopiTensorHandle_t totalWeight;
    makeTensorFromScalar(ctx, &totalWeightScalar, &totalWeight, diopi_dtype_float32, diopi_device);

    AclOpRunner<3, 2> runner("NLLLoss", ctx);
    runner.addInput(input).addInput(target).setAttr("ignore_index", ignoreIndex).addOutput(out).addOutput(totalWeight);
    if (weight) {
        runner.addInput(weight);
    } else {
        diopiTensorHandle_t weightNew;
        diopiSize_t inputShape;
        diopiGetTensorShape(input, &inputShape);
        int64_t weightDim[] = {inputShape.data[1]};
        diopiSize_t weightShape{weightDim, 1};
        diopiRequireTensor(ctx, &weightNew, &weightShape, nullptr, diopi_dtype_float32, diopi_device);
        fillTensor(ctx, &weightNew, 1.0);
        runner.addInput(weightNew);
    }
    if (reduction == diopiReduction_t::ReductionMean) {
        runner.setAttr("reduction", std::string("mean"));
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner.setAttr("reduction", std::string("sum"));
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner.setAttr("reduction", std::string("none"));
    }
    runner.run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignoreIndex) {
    auto totalWeightScalar = diopiScalar_t();
    totalWeightScalar.stype = diopi_dtype_float64;
    totalWeightScalar.fval = 0.0;
    diopiTensorHandle_t totalWeight, out;
    makeTensorFromScalar(ctx, &totalWeightScalar, &totalWeight, diopi_dtype_float32, diopi_device);

    if (reduction == diopiReduction_t::ReductionNone) {
        makeTensorLike(ctx, &out, gradOutput);
    } else {
        makeTensorFromScalar(ctx, &totalWeightScalar, &out, diopi_dtype_float32, diopi_device);
    }

    diopiTensorHandle_t weightNew;

    AclOpRunner<3, 2> runner1("NLLLoss", ctx);
    runner1.addInput(input).addInput(target).setAttr("ignore_index", ignoreIndex).addOutput(out).addOutput(totalWeight);
    if (weight) {
        runner1.addInput(weight);
    } else {
        diopiSize_t inputShape;
        diopiGetTensorShape(input, &inputShape);
        int64_t weightDim[] = {inputShape.data[1]};
        diopiSize_t weightShape{weightDim, 1};
        diopiRequireTensor(ctx, &weightNew, &weightShape, nullptr, diopi_dtype_float32, diopi_device);
        fillTensor(ctx, &weightNew, 1.0);
        runner1.addInput(weightNew);
    }
    if (reduction == diopiReduction_t::ReductionMean) {
        runner1.setAttr("reduction", std::string("mean"));
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner1.setAttr("reduction", std::string("sum"));
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner1.setAttr("reduction", std::string("none"));
    }
    runner1.run();

    AclOpRunner<5, 1> runner2("NLLLossGrad", ctx);
    runner2.addInput(input).addInput(gradOutput).addInput(target).setAttr("ignore_index", ignoreIndex).addOutput(gradInput);

    if (weight) {
        runner2.addInput(weight).addInput(totalWeight);
    } else {
        runner2.addInput(weightNew).addInput(totalWeight);
    }
    if (reduction == diopiReduction_t::ReductionMean) {
        runner2.setAttr("reduction", std::string("mean"));
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner2.setAttr("reduction", std::string("sum"));
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner2.setAttr("reduction", std::string("none"));
    }
    runner2.run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                             diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    check_args(labelSmoothing == 0, "label_smoothing %f not supported.", labelSmoothing);
    check_args(weight == nullptr, "weights not supported.");
    check_args(ignoreIndex < 0, "weights not supported.");

    diopiTensorHandle_t targetOneHot, backProp;
    diopiSize_t inputSize;

    diopiGetTensorShape(input, &inputSize);

    makeTensorLike(ctx, &targetOneHot, input);
    makeTensorLike(ctx, &backProp, input);

    diopiOneHot(ctx, targetOneHot, target, inputSize.data[1]);

    AclOpRunner<2, 2> runner("SoftmaxCrossEntropyWithLogits", ctx);
    runner.addInput(input).addInput(targetOneHot);

    if (reduction == diopiReduction_t::ReductionNone) {
        runner.addOutput(out).addOutput(backProp);
        runner.run();
    } else {
        int64_t lossDim[] = {inputSize.data[0]};
        diopiSize_t lossShape{lossDim, 1};
        diopiTensorHandle_t loss;
        diopiRequireTensor(ctx, &loss, &lossShape, nullptr, diopi_dtype_float32, diopi_device);
        runner.addOutput(loss).addOutput(backProp);
        runner.run();

        if (reduction == ReductionSum) {
            diopiSum(ctx, out, loss, diopiSize_t());
        } else if (reduction == ReductionMean) {
            diopiMean(ctx, out, loss, diopiSize_t());
        }
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    check_args(labelSmoothing == 0, "label_smoothing %f not supported.", labelSmoothing);
    check_args(weight == nullptr, "weights not supported.");
    check_args(ignoreIndex < 0, "weights not supported.");

    diopiTensorHandle_t targetOneHot;
    diopiSize_t inputSize;
    diopiGetTensorShape(input, &inputSize);
    makeTensorLike(ctx, &targetOneHot, input);
    diopiOneHot(ctx, targetOneHot, target, inputSize.data[1]);

    int64_t lossDim[] = {inputSize.data[0]};
    diopiSize_t lossShape{lossDim, 1};
    diopiTensorHandle_t loss;
    diopiRequireTensor(ctx, &loss, &lossShape, nullptr, diopi_dtype_float32, diopi_device);

    AclOpRunner<2, 2> runner("SoftmaxCrossEntropyWithLogits", ctx);
    runner.addInput(input).addInput(targetOneHot);
    runner.addOutput(loss).addOutput(gradInput);
    runner.run();

    info("reduction is %d", reduction);

    if (reduction == ReductionMean || reduction == ReductionNone) {
        auto batchSize = diopiScalar_t();
        batchSize.stype = diopi_dtype_float64;
        batchSize.fval = inputSize.data[0];
        diopiDivInpScalar(ctx, gradInput, &batchSize, RoundModeNone);
    }
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
