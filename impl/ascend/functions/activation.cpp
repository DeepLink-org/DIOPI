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
DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Relu", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    AclOpRunner<1, 1>("Relu", ctx).addInput(input).addOutput(input).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<1, 1>("SoftmaxV2", ctx).addInput(input).setAttr<int64_t>("axes", dimList).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t output, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<2, 1>("SoftmaxGrad", ctx).addInput(output).addInput(gradOutput).setAttr<int64_t>("axes", dimList).addOutput(gradInput).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<1, 1>("LogSoftmaxV2", ctx).addInput(input).setAttr("axes", dimList).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
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

DIOPI_API diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Swish", ctx).addInput(input).setAttr<float>("scale", 1.0).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiSilu(ctx, input, input); }

DIOPI_API diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                         diopiConstTensorHandle_t input) {
    diopiTensorHandle_t out;
    makeTensorLike(ctx, &out, input);
    diopiSilu(ctx, out, input);
    AclOpRunner<3, 1>("SwishGrad", ctx).addInput(gradOutput).addInput(input).addInput(out).addOutput(gradInput).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Sigmoid", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t output) {
    AclOpRunner<2, 1>("SigmoidGrad", ctx).addInput(output).addInput(gradOutput).addOutput(gradInput).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
    AclOpRunner<1, 1>("Gelu", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                         diopiConstTensorHandle_t input, const char* approximate) {
    AclOpRunner<3, 1>("GeluGrad", ctx).addInput(gradOutput).addInput(input).addInput(gradOutput).addOutput(gradInput).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negativeSlope) {
    AclOpRunner<1, 1>("LeakyRelu", ctx).addInput(input).setAttr("negative_slope", getValue<float>(negativeSlope)).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negativeSlope) {
    return diopiLeakyRelu(ctx, input, input, negativeSlope);
}

DIOPI_API diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* negativeSlope, bool inputIsResult) {
    AclOpRunner<2, 1>("LeakyReluGrad", ctx)
        .addInput(gradOutput)
        .addInput(input)
        .setAttr("negative_slope", getValue<float>(negativeSlope))
        .addOutput(gradInput)
        .run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Tanh", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                         diopiConstTensorHandle_t output) {
    AclOpRunner<2, 1>("TanhGrad", ctx).addInput(output).addInput(gradOutput).addOutput(gradInput).run();
    return diopiSuccess;
}
}
}  // namespace ascend
}  // namespace impl
