/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Relu", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

extern "C" diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    AclOpRunner<1, 1>("Relu", ctx).addInput(input).addOutput(input).run();
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<1, 1>("SoftmaxV2", ctx).addInput(input).setAttr<int64_t>("axes", dimList).addOutput(out).run();
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                       diopiConstTensorHandle_t output, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<2, 1>("SoftmaxGrad", ctx).addInput(gradOutput, output).setAttr<int64_t>("axes", dimList).addOutput(gradInput).run();
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<1, 1>("LogSoftmaxV2", ctx).addInput(input).setAttr("axes", dimList).addOutput(out).run();
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                          diopiConstTensorHandle_t output, int64_t dim) {
    std::vector<int64_t> dimList = {dim};
    AclOpRunner<2, 1>("LogSoftmaxGrad", ctx).addInput(gradOutput, output).setAttr("axes", dimList).addOutput(gradInput).run();
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Sigmoid").addInput(input).addOutput(out).run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                       diopiConstTensorHandle_t output) {
    AclOpRunner<2, 1>("SigmoidGrad").addInput(output, grad_output).addOutput(grad_input).run(ctx);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
