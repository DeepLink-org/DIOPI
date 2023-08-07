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
DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t outputSize) {
    AclOpRunner<1, 1>("AdaptiveAvgPool2d", ctx)
        .addInput(input)
        .setAttr("output_size", std::vector<int32_t>{outputSize.data[0], outputSize.data[1]})
        .addOutput(out)
        .run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                      diopiConstTensorHandle_t input) {
    diopiSize_t shape;
    diopiGetTensorShape(input, &shape);
    AclOpRunner<1, 1>("AdaptiveAvgPool2dGrad", ctx)
        .addInput(gradOutput)
        .setAttr("orig_input_shape", std::vector<int32_t>{shape.data[0], shape.data[1], shape.data[2], shape.data[3]})
        .addOutput(gradInput)
        .run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                                 diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    AclOpRunner<1, 2> runner("MaxPoolWithArgmaxV1", ctx);
    if (stride.data)
        runner.setAttr("strides", std::vector<int64_t>{1, stride.data[0], stride.data[1], 1});
    else
        runner.setAttr("strides", std::vector<int64_t>{1, kernelSize.data[0], kernelSize.data[1], 1});
    runner.addInput(input)
        .setAttr("ksize", std::vector<int64_t>{1, kernelSize.data[0], kernelSize.data[1], 1})
        .setAttr("pads", std::vector<int64_t>{1, padding.data[0], padding.data[1], 1})
        .setAttr("dilation", std::vector<int64_t>{1, dilation.data[0], dilation.data[1], 1})
        .setAttr("ceil_mode", ceilMode)
        .addOutput(out)
        .addOutput(indices, ACL_FORMAT_NC1HWC0)
        .run();

    return diopiSuccess;
}
}

}  // namespace ascend
}  // namespace impl
