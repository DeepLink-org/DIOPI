/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnRelu, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceRelu, ctx, input);
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
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<2, 1>("LogSoftmaxGrad", ctx).addInput(gradOutput).addInput(output).addOutput(gradInput).setAttr("axis", dimList).run();
    return diopiSuccess;
}

diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSilu, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSilu, ctx, input, input);
    return diopiSuccess;
}

diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSiluBackward, ctx, gradOutput, input, gradInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSigmoid, ctx, input, out);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceSigmoid, ctx, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t output) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSigmoidBackward, ctx, gradOutput, output, gradInput);
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
    DIOPI_ASCEND_CALL_ACLNN(aclnnTanh, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceTanh, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t output) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnTanhBackward, ctx, gradOutput, output, gradInput);
    return diopiSuccess;
}

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* minVal,
                           const diopiScalar_t* maxVal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnHardtanh, ctx, input, minVal, maxVal, out);
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceHardtanh, ctx, input, minVal, maxVal);
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                   const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnHardtanhBackward, ctx, gradOutput, input, minVal, maxVal, gradInput);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
