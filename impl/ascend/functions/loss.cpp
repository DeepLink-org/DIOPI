/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t nllLossOutWithTotalWeight(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t totalWeight, diopiConstTensorHandle_t input,
                                       diopiConstTensorHandle_t target, diopiTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    AscendTensor inputAt0(input), outAt0(out);

    if (0 == inputAt0.numel()) {
        // align with pytorch
        if (diopiReduction_t::ReductionMean == reduction) {
            // nan
            if (outAt0.dtype() == diopi_dtype_float16) {
                diopiTensorHandle_t outTemp;
                makeTensorLike(ctx, &outTemp, out, diopi_dtype_float64);
                diopiScalar_t nanScalar = {diopi_dtype_float64, NAN};
                diopiFill(ctx, outTemp, &nanScalar);
                diopiCastDtype(ctx, out, outTemp);
            } else {
                fillNan(ctx, outAt0);
            }
        } else if (diopiReduction_t::ReductionSum == reduction) {
            fillTensor(ctx, out, 0.0f);  // none
        } else if (diopiReduction_t::ReductionNone == reduction) {
            return diopiSuccess;  // array([])
        }
        return diopiSuccess;
    }

    diopiSize_t inputShape;
    diopiTensorHandle_t inputCopy, targetCopy;
    diopiGetTensorShape(input, &inputShape);

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);

    int64_t batch = 1;
    if (inputShape.len > 2) {
        std::vector<int64_t> inputCopyShapeVec;
        std::vector<int64_t> permuteDimVec;
        inputCopyShapeVec.push_back(inputShape.data[0]);
        permuteDimVec.push_back(0);
        for (int i = 1; i < inputShape.len - 1; i++) {
            inputCopyShapeVec.push_back(inputShape.data[i + 1]);
            permuteDimVec.push_back(i + 1);
        }
        inputCopyShapeVec.push_back(inputShape.data[1]);
        permuteDimVec.push_back(1);
        diopiSize_t inputCopyShape = vectorToDiopiSize(inputCopyShapeVec);
        diopiSize_t permuteDim = vectorToDiopiSize(permuteDimVec);

        diopiRequireTensor(ctx, &inputCopy, &inputCopyShape, nullptr, dtype, diopi_device);
        diopiPermute(ctx, inputCopy, input, permuteDim);

        for (int i = 0; i < inputCopyShapeVec.size() - 1; ++i)
            if (inputCopyShapeVec[i] != 0) batch *= inputCopyShapeVec[i];
    } else {
        inputCopy = contiguous(ctx, input);
    }

    targetCopy = contiguous(ctx, target, diopi_dtype_int32);
    AscendTensor inputAt(inputCopy), outAt(out), targetAt(targetCopy);

    AclOpRunner<3, 2> runner("NLLLoss", ctx);
    if (inputAt.dtype() != diopi_dtype_float32) {
        castTensor(ctx, inputAt, diopi_dtype_float32);
    }

    AscendTensor weightAt(weight);
    castTensor(ctx, weightAt, diopi_dtype_float32);
    if (0 <= ignoreIndex && ignoreIndex < inputAt.shape(-1)) {
        diopiStreamHandle_t stream;
        void *ptr = reinterpret_cast<uint8_t *>(const_cast<void *>(weightAt.data())) + ignoreIndex * weightAt.elemsize();
        float val = 0.0f;
        diopiGetStream(ctx, &stream);
        aclrtMemcpyAsync(ptr, sizeof(float), &val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtSynchronizeStream(stream);
    }

    // ascend only support inpu tensor with 2D dimension
    if (inputShape.len == 1) {
        reshape(ctx, inputAt, inputAt, {1, inputShape.data[0]});
        reshape(ctx, targetAt, targetAt, {targetAt.numel()});
    } else if (inputShape.len > 2) {
        reshape(ctx, inputAt, inputAt, {batch, inputShape.data[1]});
        reshape(ctx, targetAt, targetAt, {targetAt.numel()});
    }

    runner.addInput(inputAt).addInput(targetAt).addInput(weightAt).setAttr("ignore_index", ignoreIndex);

    diopiDtype_t outOriginDtype = outAt.dtype();
    if (outOriginDtype != diopi_dtype_float32) {
        castTensor(ctx, outAt, diopi_dtype_float32);
    }
    if (reduction == diopiReduction_t::ReductionMean) {
        runner.setAttr("reduction", std::string("mean"));
        runner.addOutput(outAt);
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner.setAttr("reduction", std::string("sum"));
        runner.addOutput(outAt);
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner.setAttr("reduction", std::string("none"));
        runner.addOutput(outAt);
    }
    runner.addOutput(totalWeight);
    runner.run();
    AscendTensor outOri(out);
    castTensor(ctx, outAt, outOri);

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
        // weight shape is (C). C is number of classes
        int64_t weightDim[1];
        if (inputShape.len == 1)
            weightDim[0] = inputShape.data[0];
        else
            weightDim[0] = inputShape.data[1];
        diopiSize_t weightShape = arrayToDiopiSize(weightDim, 1);
        diopiRequireTensor(ctx, &weightCopy, &weightShape, nullptr, diopi_dtype_float32, diopi_device);
        fillTensor(ctx, weightCopy, static_cast<float>(1.0));
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
        int64_t weightDim[1];
        if (inputShape.len == 1)
            weightDim[0] = inputShape.data[0];
        else
            weightDim[0] = inputShape.data[1];
        diopiSize_t weightShape = arrayToDiopiSize(weightDim, 1);
        diopiRequireTensor(ctx, &weightCopy, &weightShape, nullptr, diopi_dtype_float32, diopi_device);
        fillTensor(ctx, weightCopy, static_cast<float>(1.0));
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
    } else if (inputShape.len == 2) {
        inputCopy = contiguous(ctx, input);
        calShapeVec.push_back(inputShape.data[0]);
        calShapeVec.push_back(inputShape.data[1]);
        calTargetShapeVec.push_back(inputShape.data[0]);
    } else {  // inpusShape.len == 1
        inputCopy = contiguous(ctx, input);
        calShapeVec.push_back(1);
        calShapeVec.push_back(inputShape.data[0]);
        calTargetShapeVec.push_back(1);
    }

    void *dataPtr, *targetPtr;
    targetCopy = contiguous(ctx, target, diopi_dtype_int32);
    diopiGetTensorData(inputCopy, &dataPtr);
    diopiGetTensorData(targetCopy, &targetPtr);

    AscendTensor inputAt(inputCopy), yGradAt(gradOutput);

    if (inputAt.dtype() != diopi_dtype_float32 || yGradAt.dtype() != diopi_dtype_float32) {
        castTensor(ctx, inputAt, diopi_dtype_float32);
        castTensor(ctx, yGradAt, diopi_dtype_float32);
    }

    AclOpRunner<5, 1> runner("NLLLossGrad", ctx);

    runner.addInput(inputAt.data(), inputAt.getAclMemBufferSize(), calShapeVec, ACL_FORMAT_ND, inputAt.dtype());

    if (reduction == diopiReduction_t::ReductionMean) {
        runner.setAttr("reduction", std::string("mean"));
        runner.addInput(yGradAt);
    } else if (reduction == diopiReduction_t::ReductionSum) {
        runner.setAttr("reduction", std::string("sum"));
        runner.addInput(yGradAt);
    } else if (reduction == diopiReduction_t::ReductionNone) {
        runner.setAttr("reduction", std::string("none"));
        runner.addInput(yGradAt.data(), yGradAt.getAclMemBufferSize(), calTargetShapeVec, ACL_FORMAT_ND, yGradAt.dtype());
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
