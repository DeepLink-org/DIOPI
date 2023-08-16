/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>
#include <vector>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    auto format = getAclDataFormat(input);
    const std::string dataFormat = (format == ACL_FORMAT_NHWC) ? "NHWC" : "NCHW";
    std::vector<int64_t> strideTemp(4, 1);
    std::vector<int64_t> dilationsTemp(4, 1);
    if (format == ACL_FORMAT_NHWC) {
        strideTemp[1] = stride.data[0];
        strideTemp[2] = stride.data[1];
        dilationsTemp[1] = dilation.data[0];
        dilationsTemp[2] = dilation.data[1];
    } else {
        strideTemp[2] = stride.data[0];
        strideTemp[3] = stride.data[1];
        dilationsTemp[2] = dilation.data[0];
        dilationsTemp[3] = dilation.data[1];
    }
    const std::vector<int64_t> paddingTemp = {padding.data[0], padding.data[0], padding.data[1], padding.data[1]};

    AclOpRunner<3, 1> runner("Conv2D", ctx);
    runner.addInput(input, weight)
        .setAttr("strides", strideTemp)
        .setAttr("pads", paddingTemp)
        .setAttr("dilations", dilationsTemp)
        .setAttr<int32_t>("groups", groups)
        .setAttr("data_format", dataFormat)
        .addOutput(out);
    if (bias) {
        runner.addInput(bias);
    }
    runner.run();
    return diopiSuccess;
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                        diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                        diopiSize_t *biasSizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    auto format = getAclDataFormat(input);
    const std::string dataFormat = (format == ACL_FORMAT_NHWC) ? "NHWC" : "NCHW";
    std::vector<int64_t> strideTemp(4, 1);
    std::vector<int64_t> dilationsTemp(4, 1);
    if (format == ACL_FORMAT_NHWC) {
        strideTemp[1] = stride.data[0];
        strideTemp[2] = stride.data[1];
        dilationsTemp[1] = dilation.data[0];
        dilationsTemp[2] = dilation.data[1];
    } else {
        strideTemp[2] = stride.data[0];
        strideTemp[3] = stride.data[1];
        dilationsTemp[2] = dilation.data[0];
        dilationsTemp[3] = dilation.data[1];
    }
    const std::vector<int64_t> paddingTemp = {padding.data[0], padding.data[0], padding.data[1], padding.data[1]};

    diopiSize_t weightShape;
    diopiGetTensorShape(gradWeight, &weightShape);

    AclOpRunner<3, 1>("Conv2DBackpropFilter", ctx)
        .addInput(input)
        .addConstInput(weightShape)
        .addInput(gradOutput)
        .addOutput(gradWeight)
        .setAttr("strides", strideTemp)
        .setAttr("pads", paddingTemp)
        .setAttr("dilations", dilationsTemp)
        .setAttr("groups", groups)
        .setAttr("data_format", dataFormat)
        .run();
    if (gradInput != nullptr) {
        diopiSize_t inputShape;
        diopiGetTensorShape(input, &inputShape);
        AclOpRunner<3, 1>("Conv2DBackpropInput", ctx)
            .addConstInput(inputShape)
            .addInput(weight)
            .addInput(gradOutput)
            .addOutput(gradInput)
            .setAttr("strides", strideTemp)
            .setAttr("pads", paddingTemp)
            .setAttr("dilations", dilationsTemp)
            .setAttr("data_format", dataFormat)
            .setAttr("groups", groups)
            .run();
    }

    if (gradBias != nullptr) {
        AclOpRunner<1, 1>("BiasAddGrad", ctx).addInput(gradOutput).addOutput(gradBias).setAttr("data_format", dataFormat).run();
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
