/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

extern "C" {
#if 0
diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    BEGIN_CALL_ACL_OP(out, indices, input, kernel_size, stride, padding, dilation);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(indicesAt);
    //std::tie(outAt, indicesAt) =
    //acl_op::max_pool2d_with_indices_out(inputAt, kernel_sizeAt, strideAt, paddingAt, dilationAt, ceil_mode, outAt, indicesAt);
    acl_op::max_pool2d_with_indices(inputAt, kernel_sizeAt, strideAt, paddingAt, dilationAt, ceil_mode);
    END_CALL_ACL_OP();
}

#else

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    BEGIN_CALL_ACL_OP(out, indices, input, kernel_size, stride, padding, dilation);
    at_npu::native::OpCommand cmd;
    #if 0
    cmd.Name("MaxPoolV3")
      .Input(inputAt, "x")
      .Output(outAt, "y")
      .Attr("ksize", kernel_sizeAt)
      .Attr("strides", strideAt)
      .Attr("padding_mode", "CALCULATED")
      .Attr("pads", paddingAt)
      .Attr("data_format", "NHWC")
      .Attr("ceil_mode", ceil_mode)
      .Run();
    cmd.Name("MaxPoolWithArgmaxV1")
        .Input(inputAt, "x")
        .Output(outAt, "y")
        .Output(indicesAt, "argmax", c10::nullopt, "uint16")
        .Attr("ksize", kernel_sizeAt)
        .Attr("strides", strideAt)
        .Attr("pads", paddingAt)
        .Attr("dilation", dilationAt)
        .Attr("ceil_mode", ceil_mode)
        .Run();
    #endif
    auto indice_ = indicesAt.to(at::ScalarType::Short);
    cmd.Name("MaxPoolWithArgmaxV2")
        .Input(inputAt, "x")
        .Output(outAt, "y")
        .Output(indice_, "argmax", c10::nullopt, "uint16")
        .Attr("ksize", kernel_sizeAt)
        .Attr("strides", strideAt)
        .Attr("pads", paddingAt)
        .Attr("dilation", dilationAt)
        .Attr("ceil_mode", ceil_mode)
        .Run();
    END_CALL_ACL_OP();
}
#endif


diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                    bool ceil_mode, diopiConstTensorHandle_t indices) {
    BEGIN_CALL_ACL_OP(grad_input, grad_output, input, kernel_size, stride, padding, dilation, indices);
    //at_npu::native::OpPreparation::markAsOutputForApplyTensor(grad_inputAt);
    grad_inputAt = acl_op::max_pool2d_with_indices_backward(grad_outputAt, inputAt, kernel_sizeAt, strideAt, paddingAt, dilationAt, ceil_mode, indicesAt);
    END_CALL_ACL_OP();
}

}  // extern "C"