/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

extern "C" {

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    BEGIN_CALL_ACL_OP(out, indices, input, kernel_size, stride, padding, dilation);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(indicesAt);
    std::tie(outAt, indicesAt) = acl_op::max_pool2d_with_indices(inputAt, kernel_sizeAt, strideAt, paddingAt, dilationAt, ceil_mode);
    END_CALL_ACL_OP();
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                    bool ceil_mode, diopiConstTensorHandle_t indices) {
    BEGIN_CALL_ACL_OP(grad_input, grad_output, input, kernel_size, stride, padding, dilation, indices);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(grad_inputAt);
    grad_inputAt = acl_op::max_pool2d_with_indices_backward(grad_outputAt, inputAt, kernel_sizeAt, strideAt, paddingAt, dilationAt, ceil_mode, indicesAt);
    END_CALL_ACL_OP();
}

}  // extern "C"