/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
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

    AclOpRunner<3, 1> runner("Conv2D");
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
    runner.run(ctx);
    return diopiSuccess;
}

extern "C" diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                   diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                   diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                   diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
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

    diopiTensorHandle_t weightShapeT;
    diopiSize_t weightShape;
    diopiGetTensorShape(grad_weight, &weightShape);
    makeTensorFromSize<int32_t>(ctx, &weightShape, &weightShapeT, diopi_dtype_int32);

    AclOpRunner<3, 1>("Conv2DBackpropFilter")
        .addInput<0>(input)
        .addConstInput<1>(weightShapeT, ACL_FORMAT_ND)
        .addInput<2>(grad_output)
        .addOutput(grad_weight)
        .setAttr("strides", strideTemp)
        .setAttr("pads", paddingTemp)
        .setAttr("dilations", dilationsTemp)
        .setAttr("groups", groups)
        .setAttr("data_format", dataFormat)
        .run(ctx);
    if (grad_input != nullptr) {
        diopiTensorHandle_t inputShapeT;
        diopiSize_t inputShape;
        diopiGetTensorShape(input, &inputShape);
        makeTensorFromSize<int32_t>(ctx, &inputShape, &inputShapeT, diopi_dtype_int32);
        AclOpRunner<3, 1>("Conv2DBackpropInput")
            .addConstInput<0>(inputShapeT)
            .addInput<1>(weight)
            .addInput<2>(grad_output)
            .addOutput(grad_input)
            .setAttr("strides", strideTemp)
            .setAttr("pads", paddingTemp)
            .setAttr("dilations", dilationsTemp)
            .setAttr("data_format", dataFormat)
            .setAttr("groups", groups)
            .run(ctx);
    }

    if (grad_bias != nullptr) {
        AclOpRunner<1, 1>("BiasAddGrad").addInput(grad_output).addOutput(grad_bias).setAttr("data_format", dataFormat).run(ctx);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
