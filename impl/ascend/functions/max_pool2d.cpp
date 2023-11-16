/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"


namespace impl {
namespace ascend {
diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                            diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    AscendTensor inputTemp(input);
    AscendTensor outputTemp(out);

    std::cout << "inputTemp.dtype = " << inputTemp.dtype();
    std::cout << "outputTemp.dtype = " << outputTemp.dtype();

    if (inputTemp.dim() == 3) {
        inputTemp.unsqueeze(0);
        outputTemp.unsqueeze(0);
    }

    auto format = getAclDataFormat(input);
    const std::string dataFormat = (format == ACL_FORMAT_NHWC) ? "NHWC" : "NCHW";

    std::vector<int64_t> strideTemp(4, 1);
    std::vector<int64_t> ksizeTemp(4, 1);
    std::vector<int64_t> paddingTemp(4, 1);
    std::vector<int64_t> dilationTemp(4, 1);

    const int64_t kernelH = kernel_size.data[0];
    const int64_t kernelW = kernel_size.len == 1 ? kernelH : kernel_size.data[1];

    int64_t channelH;
    int64_t channelW;

    // NHWC
    if (format == ACL_FORMAT_NHWC) {
        channelH = 1;
        channelW = 2;
    // NCHW
    } else {
        channelH = 2;
        channelW = 3;
    }

    // stride setting
    if (stride.len == 0) {
        strideTemp[channelH] == kernelH;
        strideTemp[channelW] == kernelW;
    } else {
        if (stride.len == 1) {
            strideTemp[channelH] = stride.data[0];
            strideTemp[channelW] = stride.data[0];
        } else {
            strideTemp[channelH] = stride.data[0];
            strideTemp[channelW] = stride.data[1];
        }
    }
    // ASCEND_CHECK_ABORT(strideTemp[channelH] <= 63 && strideTemp[channelW] <= 63, "strides should be less than or equal to 63");

    // dialtion setting
    if (dilation.len == 1) {
        dilationTemp[channelH] = dilation.data[0];
        dilationTemp[channelW] = dilation.data[0];
    } else if (dilation.len == 2) {
        dilationTemp[channelH] = dilation.data[0];
        dilationTemp[channelW] = dilation.data[1];
    }

    // padding setting
    if (padding.len == 1) {
        paddingTemp[channelH] = padding.data[0];
        paddingTemp[channelW] = padding.data[0];
    } else if (padding.len == 2) {
        paddingTemp[channelH] = padding.data[0];
        paddingTemp[channelW] = padding.data[1];
    }

    // kernel_size setting
    ksizeTemp[channelH] = kernelH;
    ksizeTemp[channelW] = kernelW;

    AclOpRunner<1, 1> runner("MaxPoolV3", ctx);
    runner.addInput(inputTemp)
        .setAttr("ksize", ksizeTemp)
        .setAttr("strides", strideTemp)
        .setAttr("padding_mode", std::string{"CALCULATED"})
        .setAttr("pads", paddingTemp)
        .setAttr("data_format", dataFormat)
        .setAttr("global_pooling", false)
        .setAttr("dilations", dilationTemp)
        .setAttr("ceil_mode", ceil_mode)
        .addOutput(outputTemp);
    runner.run();
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                    bool ceil_mode, diopiConstTensorHandle_t indices) {
    AscendTensor inputTemp(input);
    if (inputTemp.dim() == 3) {
        inputTemp.unsqueeze(0);
    }

    auto format = getAclDataFormat(input);
    const std::string dataFormat = (format == ACL_FORMAT_NHWC) ? "NHWC" : "NCHW";

    std::vector<int64_t> strideTemp(4, 1);
    std::vector<int64_t> ksizeTemp(4, 1);
    std::vector<int64_t> paddingTemp(4, 1);
    std::vector<int64_t> dilationTemp(4, 1);

    const int64_t kernelH = kernel_size.data[0];
    const int64_t kernelW = kernel_size.len == 1 ? kernelH : kernel_size.data[1];

    int64_t channelH;
    int64_t channelW;

    if (format == ACL_FORMAT_NHWC) {
        channelH = 1;
        channelW = 2;
    } else {
        channelH = 2;
        channelW = 3;
    }

    // stride setting
    if (stride.len == 0) {
        strideTemp[channelH] == kernelH;
        strideTemp[channelW] == kernelW;
    } else {
        if (stride.len == 1) {
            strideTemp[channelH] = stride.data[0];
            strideTemp[channelW] = stride.data[0];
        } else {
            strideTemp[channelH] = stride.data[0];
            strideTemp[channelW] = stride.data[1];
        }
    }

    // dialtion setting
    if (dilation.len == 1) {
        dilationTemp[channelH] = dilation.data[0];
        dilationTemp[channelW] = dilation.data[0];
    } else if (dilation.len == 2) {
        dilationTemp[channelH] = dilation.data[0];
        dilationTemp[channelW] = dilation.data[1];
    }

    // padding setting
    if (padding.len == 1) {
        paddingTemp[channelH] = padding.data[0];
        paddingTemp[channelW] = padding.data[0];
    } else if (padding.len == 2) {
        paddingTemp[channelH] = padding.data[0];
        paddingTemp[channelW] = padding.data[1];
    }

    // kernel_size setting
    ksizeTemp[channelH] = kernelH;
    ksizeTemp[channelW] = kernelW;

    AclOpRunner<1, 1> runner("MaxPoolGradWithArgmax", ctx);
    runner.addInput(inputTemp)
        .addInput(grad_output)
        .addInput(indices)
        .setAttr("ksize", ksizeTemp)
        .setAttr("strides", strideTemp)
        .setAttr("padding_mode", std::string{"CALCULATED"})
        // .setAttr("pads", paddingTemp)
        // .setAttr("data_format", dataFormat)
        // .setAttr("global_pooling", false)
        // .setAttr("dilations", dilationTemp)
        // .setAttr("ceil_mode", ceil_mode)
        .addOutput(grad_input);
    runner.run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl