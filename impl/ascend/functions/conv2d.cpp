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
    runner.addInput(input)
        .addInput(weight)
        .setAttr("strides", strideTemp)
        .setAttr("pads", paddingTemp)
        .setAttr("dilations", dilationsTemp)
        .setAttr<int64_t>("groups", 1)
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

    gradOutput = contiguous(ctx, gradOutput);
    auto gradOutFormat = getAclDataFormat(gradOutput);

    diopiTensorHandle_t gradOutputCopy;
    if (gradOutFormat == ACL_FORMAT_NHWC) {
        diopiSize_t gradOutputSize;
        diopiDtype_t dtype;
        diopiGetTensorShape(gradOutput, &gradOutputSize);
        diopiGetTensorDtype(gradOutput, &dtype);
        diopiRequireTensor(ctx, &gradOutputCopy, &gradOutputSize, nullptr, dtype, diopi_device);
        AclOpRunner<1, 1>("TransData", ctx)
            .addInput(gradOutput, ACL_FORMAT_NHWC)
            .setAttr("src_format", std::string("NHWC"))
            .setAttr("dst_format", std::string("NCHW"))
            .addOutput(gradOutputCopy, ACL_FORMAT_NCHW)
            .run();
    } else {
        gradOutputCopy = const_cast<diopiTensorHandle_t>(gradOutput);
    }

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
        .addConstInput(weightShape, diopi_dtype_int32)
        .addInput(gradOutputCopy)
        .addOutput(gradWeight)
        .setAttr("strides", strideTemp)
        .setAttr("pads", paddingTemp)
        .setAttr("dilations", dilationsTemp)
        .setAttr<int64_t>("groups", 1)
        .setAttr("data_format", dataFormat)
        .run();
    if (gradInput != nullptr) {
        diopiSize_t inputShape;
        diopiGetTensorShape(input, &inputShape);
        AclOpRunner<3, 1>("Conv2DBackpropInput", ctx)
            .addConstInput(inputShape, diopi_dtype_int32)
            .addInput(weight)
            .addInput(gradOutputCopy)
            .addOutput(gradInput)
            .setAttr("strides", strideTemp)
            .setAttr("pads", paddingTemp)
            .setAttr("dilations", dilationsTemp)
            .setAttr("data_format", dataFormat)
            .setAttr<int64_t>("groups", 1)
            .run();
    }

    if (gradBias != nullptr) {
        AclOpRunner<1, 1>("BiasAddGrad", ctx).addInput(gradOutputCopy).addOutput(gradBias).setAttr("data_format", dataFormat).run();
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
