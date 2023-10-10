/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Relu", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    AclOpRunner<1, 1>("Relu", ctx).addInput(input).addOutput(input).run();
    return diopiSuccess;
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<1, 1>("SoftmaxV2", ctx).addInput(input).setAttr<int64_t>("axes", dimList).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t output,
                                  int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<2, 1>("SoftmaxGrad", ctx).addInput(output).addInput(gradOutput).setAttr<int64_t>("axes", dimList).addOutput(gradInput).run();
    return diopiSuccess;
}

diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<1, 1>("LogSoftmaxV2", ctx).addInput(input).setAttr("axes", dimList).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                     diopiConstTensorHandle_t output, int64_t dim) {
    diopiSize_t sumSize;
    diopiGetTensorShape(gradOutput, &sumSize);
    std::vector<int64_t> sumSizeVec(sumSize.data, sumSize.data + sumSize.len);
    if (dim < 0) dim += sumSize.len;
    sumSizeVec[dim] = 1;
    sumSize = vectorToDiopiSize(sumSizeVec);
    diopiTensorHandle_t sum, exp;
    diopiDtype_t dtype;
    diopiGetTensorDtype(gradOutput, &dtype);
    diopiRequireTensor(ctx, &sum, &sumSize, nullptr, dtype, diopi_device);
    std::vector<int64_t> dimVec({dim});
    auto dimSize = vectorToDiopiSize(dimVec);
    diopiSum(ctx, sum, gradOutput, dimSize);
    makeTensorLike(ctx, &exp, output);
    diopiExp(ctx, exp, output);
    diopiMul(ctx, gradInput, exp, sum);
    diopiScalar_t scalar;
    scalar.stype = diopi_dtype_float64;
    scalar.fval = 1.0;
    diopiSub(ctx, gradInput, gradOutput, gradInput, &scalar);
    return diopiSuccess;
}

diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Swish", ctx).addInput(input).setAttr<float>("scale", 1.0).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiSilu(ctx, input, input); }

diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input) {
    diopiTensorHandle_t out;
    makeTensorLike(ctx, &out, input);
    diopiSilu(ctx, out, input);
    AclOpRunner<3, 1>("SwishGrad", ctx).addInput(gradOutput).addInput(input).addInput(out).addOutput(gradInput).run();
    return diopiSuccess;
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Sigmoid", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiSigmoid(ctx, input, input); }

DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t output) {
    AclOpRunner<2, 1>("SigmoidGrad", ctx).addInput(output).addInput(gradOutput).addOutput(gradInput).run();
    return diopiSuccess;
}

diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
    AclOpRunner<1, 1>("Gelu", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                               const char* approximate) {
    AclOpRunner<3, 1>("GeluGrad", ctx).addInput(gradOutput).addInput(input).addInput(gradOutput).addOutput(gradInput).run();
    return diopiSuccess;
}

diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negativeSlope) {
    AclOpRunner<1, 1>("LeakyRelu", ctx).addInput(input).setAttr("negative_slope", getValue<float>(negativeSlope)).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negativeSlope) {
    return diopiLeakyRelu(ctx, input, input, negativeSlope);
}

diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* negativeSlope, bool inputIsResult) {
    AclOpRunner<2, 1>("LeakyReluGrad", ctx)
        .addInput(gradOutput)
        .addInput(input)
        .setAttr("negative_slope", getValue<float>(negativeSlope))
        .addOutput(gradInput)
        .run();
    return diopiSuccess;
}

diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Tanh", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t output) {
    AclOpRunner<2, 1>("TanhGrad", ctx).addInput(output).addInput(gradOutput).addOutput(gradInput).run();
    return diopiSuccess;
}

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* minVal,
                           const diopiScalar_t* maxVal) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiScalar_t min = *minVal;
    if (isIntegralTypeWithBool(dtype)) {
        if (maxVal->ival < minVal->ival) {
            min = *maxVal;
        }
    } else {
        if (maxVal->fval < minVal->fval) {
            min = *maxVal;
        }
    }
    AclOpRunner<3, 1>("ClipByValue", ctx).addInput(input).addConstInput(min, dtype).addConstInput(*maxVal, dtype).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    return diopiHardtanh(ctx, input, input, minVal, maxVal);
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                   const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    diopiDtype_t inDtype;
    diopiGetTensorDtype(input, &inDtype);
    std::set<diopiDtype_t> supportDtypes{diopi_dtype_float16, diopi_dtype_float32};
    if (!supportDtypes.count(inDtype)) {
        diopiTensorHandle_t tmpInput;
        makeTensorLike(ctx, &tmpInput, input, diopi_dtype_float32);
        diopiCastDtype(ctx, tmpInput, input);

        diopiTensorHandle_t tmpGradOutput;
        makeTensorLike(ctx, &tmpGradOutput, gradOutput, diopi_dtype_float32);
        diopiCastDtype(ctx, tmpGradOutput, gradOutput);

        AclOpRunner<2, 1>("HardtanhGrad", ctx)
            .addInput(tmpInput)
            .addInput(tmpGradOutput)
            .addOutput(gradInput)
            .setAttr("max_val", getValue<float>(maxVal))
            .setAttr("min_val", getValue<float>(minVal))
            .run();
    } else {
        AclOpRunner<2, 1>("HardtanhGrad", ctx)
            .addInput(input)
            .addInput(gradOutput)
            .addOutput(gradInput)
            .setAttr("max_val", getValue<float>(maxVal))
            .setAttr("min_val", getValue<float>(minVal))
            .run();
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
