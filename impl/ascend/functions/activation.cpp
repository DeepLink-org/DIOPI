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
    AclOpRunner<1, 1>("Relu").addInput(input).addOutput(out).run(ctx);
    return diopiSuccess;
}

extern "C" diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    AclOpRunner<1, 1>("Relu").addInput(input).addOutput(input).run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dim_list = {dim};
    AclOpRunner<1, 1>("SoftmaxV2").addInput(input).setAttr<int64_t>("axes", dim_list).addOutput(out).run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                       diopiConstTensorHandle_t output, int64_t dim) {
    std::vector<int64_t> dim_list = {dim};
    AclOpRunner<2, 1>("SoftmaxGrad").addInput(grad_output, output).setAttr<int64_t>("axes", dim_list).addOutput(grad_input).run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dim_list = {dim};
    AclOpRunner<1, 1>("LogSoftmaxV2").addInput(input).setAttr("axes", dim_list).addOutput(out).run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                          diopiConstTensorHandle_t output, int64_t dim) {
    std::vector<int64_t> dim_list = {dim};
    AclOpRunner<2, 1>("LogSoftmaxGrad").addInput(grad_output, output).addInput(output).setAttr("axes", dim_list).addOutput(grad_input).run(ctx);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
