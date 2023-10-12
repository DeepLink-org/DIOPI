/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t nllLossOutWithTotalWeight(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t totalWeight, diopiConstTensorHandle_t input,
                                       diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    diopiSize_t inputShape;
    diopiTensorHandle_t inputCopy, targetCopy;
    diopiGetTensorShape(input, &inputShape);

    std::vector<int64_t> calShapeVec;
    std::vector<int64_t> calTargetShapeVec;

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);

    if (inputShape.len > 2) {
        int64_t calShape0 = inputShape.data[0];
        std::vector<int64_t> inputCopyShapeVec;
        std::vector<int64_t> permuteDimVec;
        inputCopyShapeVec.push_back(inputShape.data[0]);
        permuteDimVec.push_back(0);
        for (int i = 1; i < inputShape.len - 1; i++) {
            inputCopyShapeVec.push_back(inputShape.data[i + 1]);
            permuteDimVec.push_back(i + 1);
            calShape0 *= inputShape.data[i + 1];
        }
        calShapeVec.push_back(calShape0);
        calTargetShapeVec.push_back(calShape0);
        inputCopyShapeVec.push_back(inputShape.data[1]);
        calShapeVec.push_back(inputShape.data[1]);
        permuteDimVec.push_back(1);
        diopiSize_t inputCopyShape = vectorToDiopiSize(inputCopyShapeVec);
        diopiSize_t permuteDim = vectorToDiopiSize(permuteDimVec);

        diopiRequireTensor(ctx, &inputCopy, &inputCopyShape, nullptr, dtype, diopi_device);
        diopiPermute(ctx, inputCopy, input, permuteDim);
    } else {
        inputCopy = contiguous(ctx, input);
        calShapeVec.push_back(inputShape.data[0]);
        calShapeVec.push_back(inputShape.data[1]);
        calTargetShapeVec.push_back(inputShape.data[0]);
    }

    void *dataPtr, *targetPtr;
    targetCopy = contiguous(ctx, target, diopi_dtype_int32);
    diopiGetTensorData(inputCopy, &dataPtr);
    diopiGetTensorData(targetCopy, &targetPtr);

    AclOpRunner<3, 2> runner("NLLLoss", ctx);
    runner.addInput(dataPtr, getBaseBufferSize(inputCopy), calShapeVec, ACL_FORMAT_ND, dtype)
        .addInput(targetPtr, getBaseBufferSize(targetCopy), calTargetShapeVec, ACL_FORMAT_ND, diopi_dtype_int32)
        .setAttr("ignore_index", ignoreIndex);

    runner.addInput(weight, diopi_dtype_float32);

    if (reduction == diopiReduction_t::ReductionMean) {
        runner.setAttr("reduction", std::string("mean"));
        runner.addOutput(out);
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner.setAttr("reduction", std::string("sum"));
        runner.addOutput(out);
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner.setAttr("reduction", std::string("none"));
        diopiDtype_t outDtype;
        void *outPtr;
        diopiGetTensorDtype(out, &outDtype);
        diopiGetTensorData(out, &outPtr);
        runner.addOutput(outPtr, getBaseBufferSize(out), calTargetShapeVec, ACL_FORMAT_ND, outDtype);
    }
    runner.addOutput(totalWeight);
    runner.run();
    return diopiSuccess;
}

std::string getReductionStr(const diopiReduction_t reduction) {
    std::string reductionStr = "none";
    if (diopiReduction_t::ReductionMean == reduction) {
        reductionStr = "mean";
    } else if (diopiReduction_t::ReductionSum == reduction) {
        reductionStr = "sum";
    } else if (diopiReduction_t::ReductionEND == reduction) {
        reductionStr = "end";
    }
    return reductionStr;
}

diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    auto totalWeightSizeVec = std::vector<int64_t>({1});
    auto totalWeightSize = vectorToDiopiSize(totalWeightSizeVec);
    diopiTensorHandle_t totalWeight, weightCopy;
    diopiRequireTensor(ctx, &totalWeight, &totalWeightSize, nullptr, diopi_dtype_float32, diopi_device);

    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);

    if (weight) {
        weightCopy = contiguous(ctx, weight, diopi_dtype_float32);
    } else {
        int64_t weightDim[] = {inputShape.data[1]};
        diopiSize_t weightShape = arrayToDiopiSize(weightDim, 1);
        diopiRequireTensor(ctx, &weightCopy, &weightShape, nullptr, diopi_dtype_float32, diopi_device);
        fillTensor(ctx, &weightCopy, 1.0);
    }

    nllLossOutWithTotalWeight(ctx, out, totalWeight, input, target, weightCopy, reduction, ignoreIndex);
    return diopiSuccess;
}

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    auto totalWeightSizeVec = std::vector<int64_t>({1});
    auto totalWeightSize = vectorToDiopiSize(totalWeightSizeVec);
    diopiTensorHandle_t weightCopy, totalWeight, out, inputCopy, targetCopy, gradInputCopy;
    diopiRequireTensor(ctx, &totalWeight, &totalWeightSize, nullptr, diopi_dtype_float32, diopi_device);
    makeTensorLike(ctx, &out, gradOutput);

    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);

    if (weight) {
        weightCopy = contiguous(ctx, weight, diopi_dtype_float32);
    } else {
        int64_t weightDim[] = {inputShape.data[1]};
        diopiSize_t weightShape = arrayToDiopiSize(weightDim, 1);
        diopiRequireTensor(ctx, &weightCopy, &weightShape, nullptr, diopi_dtype_float32, diopi_device);
        fillTensor(ctx, &weightCopy, 1.0);
    }

    nllLossOutWithTotalWeight(ctx, out, totalWeight, input, target, weightCopy, reduction, ignoreIndex);

    std::vector<int64_t> calShapeVec;
    std::vector<int64_t> calTargetShapeVec;

    diopiDtype_t dtype, gradDtype;
    diopiGetTensorDtype(input, &dtype);
    diopiGetTensorDtype(gradInput, &gradDtype);

    if (inputShape.len > 2) {
        int64_t calShape0 = inputShape.data[0];
        std::vector<int64_t> inputCopyShapeVec;
        std::vector<int64_t> permuteDimVec;
        inputCopyShapeVec.push_back(inputShape.data[0]);
        permuteDimVec.push_back(0);
        for (int i = 1; i < inputShape.len - 1; i++) {
            inputCopyShapeVec.push_back(inputShape.data[i + 1]);
            permuteDimVec.push_back(i + 1);
            calShape0 *= inputShape.data[i + 1];
        }
        calShapeVec.push_back(calShape0);
        calTargetShapeVec.push_back(calShape0);
        inputCopyShapeVec.push_back(inputShape.data[1]);
        calShapeVec.push_back(inputShape.data[1]);
        permuteDimVec.push_back(1);
        diopiSize_t inputCopyShape = vectorToDiopiSize(inputCopyShapeVec);
        diopiSize_t permuteDim = vectorToDiopiSize(permuteDimVec);

        diopiRequireTensor(ctx, &inputCopy, &inputCopyShape, nullptr, dtype, diopi_device);
        diopiPermute(ctx, inputCopy, input, permuteDim);
        diopiRequireTensor(ctx, &gradInputCopy, &inputCopyShape, nullptr, dtype, diopi_device);
    } else {
        inputCopy = contiguous(ctx, input);
        calShapeVec.push_back(inputShape.data[0]);
        calShapeVec.push_back(inputShape.data[1]);
        calTargetShapeVec.push_back(inputShape.data[0]);
    }

    void *dataPtr, *targetPtr;
    targetCopy = contiguous(ctx, target, diopi_dtype_int32);
    diopiGetTensorData(inputCopy, &dataPtr);
    diopiGetTensorData(targetCopy, &targetPtr);

    AclOpRunner<5, 1> runner("NLLLossGrad", ctx);

    runner.addInput(dataPtr, getBaseBufferSize(inputCopy), calShapeVec, ACL_FORMAT_ND, dtype);

    if (reduction == diopiReduction_t::ReductionMean) {
        runner.setAttr("reduction", std::string("mean"));
        runner.addInput(gradOutput);
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner.setAttr("reduction", std::string("sum"));
        runner.addInput(gradOutput);
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner.setAttr("reduction", std::string("none"));
        auto gradOutputCopy = contiguous(ctx, gradOutput);
        void *gradOutputPtr;
        diopiGetTensorData(gradOutputCopy, &gradOutputPtr);
        runner.addInput(gradOutputPtr, getBaseBufferSize(gradOutputCopy), calTargetShapeVec, ACL_FORMAT_ND, gradDtype);
    }

    runner.addInput(targetPtr, getBaseBufferSize(targetCopy), calTargetShapeVec, ACL_FORMAT_ND, diopi_dtype_int32).setAttr("ignore_index", ignoreIndex);

    if (inputShape.len > 2) {
        void *gradInputPtr;
        diopiGetTensorData(gradInputCopy, &gradInputPtr);
        runner.addOutput(gradInputPtr, getBaseBufferSize(gradInputCopy), calShapeVec, ACL_FORMAT_ND, gradDtype);
    } else {
        runner.addOutput(gradInput);
    }
    runner.addInput(weightCopy).addInput(totalWeight);
    runner.run();

    if (inputShape.len > 2) {
        std::vector<int64_t> permuteDimVec;
        permuteDimVec.push_back(0);
        permuteDimVec.push_back(inputShape.len - 1);
        for (int i = 1; i < inputShape.len - 1; i++) {
            permuteDimVec.push_back(i);
        }
        diopiSize_t permuteDim = vectorToDiopiSize(permuteDimVec);
        diopiPermute(ctx, gradInput, gradInputCopy, permuteDim);
    }
    return diopiSuccess;
}

diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                   diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    diopiTensorHandle_t logTensor;
    makeTensorLike(ctx, &logTensor, input);
    diopiLogSoftmax(ctx, logTensor, input, 1);
    target = hostToDevice(ctx, target);
    diopiNLLLoss(ctx, out, logTensor, target, weight, reduction, ignoreIndex);
    return diopiSuccess;
}

diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                           diopiReduction_t reduction, int64_t ignoreIndex, double labelSmoothing) {
    diopiTensorHandle_t logTensor, gradLog;
    makeTensorLike(ctx, &logTensor, input);
    diopiLogSoftmax(ctx, logTensor, input, 1);
    makeTensorLike(ctx, &gradLog, gradInput);
    target = hostToDevice(ctx, target);
    diopiNLLLossBackward(ctx, gradLog, gradOutput, input, target, weight, reduction, ignoreIndex);
    diopiLogSoftmaxBackward(ctx, gradInput, gradLog, logTensor, 1);
    return diopiSuccess;
}

diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiReduction_t reduction) {
    AclOpRunner<2, 1>("MseLoss", ctx).addInput(input).addInput(target).addOutput(out).setAttr<std::string>("reduction", getReductionStr(reduction)).run();
    return diopiSuccess;
}

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    AclOpRunner<3, 1>("MseLossGrad", ctx)
        .addInput(input)
        .addInput(target)
        .addInput(gradOutput)
        .addOutput(gradInput)
        .setAttr<std::string>("reduction", getReductionStr(reduction))
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
