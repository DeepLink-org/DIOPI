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
    if (stride.len > 0 && stride.data)
        runner.setAttr("strides", std::vector<int64_t>{1, stride.data[0], stride.data[1], 1});
    else
        runner.setAttr("strides", std::vector<int64_t>{1, kernelSize.data[0], kernelSize.data[1], 1});
    diopiTensorHandle_t indicesUint16;
    diopiSize_t indicesShape;
    diopiGetTensorShape(indices, &indicesShape);
    std::vector<int64_t> indicesShapeVec(indicesShape.data, indicesShape.data + indicesShape.len);
    std::vector<int64_t> indicesShapeTmpVec = indicesShapeVec;
    indicesShapeTmpVec[1] = indicesShapeVec[1] / 16;
    indicesShapeTmpVec.push_back(16);
    diopiRequireTensor(ctx, &indicesUint16, &indicesShape, nullptr, diopi_dtype_int16, diopi_device);
    void *indicesUint16Ptr;
    diopiGetTensorData(indicesUint16, &indicesUint16Ptr);
    runner.addInput(input)
        .setAttr("ksize", std::vector<int64_t>{1, kernelSize.data[0], kernelSize.data[1], 1})
        .setAttr("pads", std::vector<int64_t>{1, padding.data[0], padding.data[1], 1})
        .setAttr("dilation", std::vector<int64_t>{1, dilation.data[0], dilation.data[1], 1})
        .setAttr("ceil_mode", ceilMode)
        .addOutput(out)
        .addOutput(indicesUint16Ptr, getBaseBufferSize(indicesUint16), indicesShapeTmpVec, ACL_FORMAT_NC1HWC0, diopi_dtype_uint16)
        .run();
    AclOpRunner<1, 1>("Cast", ctx)
        .addInput(indicesUint16Ptr, getBaseBufferSize(indicesUint16), indicesShapeVec, ACL_FORMAT_NCHW, diopi_dtype_uint16)
        .addOutput(indices)
        .setAttr<int32_t>("dst_type", getAclDataType(indices))
        .run();
    return diopiSuccess;
}
}

}  // namespace ascend
}  // namespace impl
