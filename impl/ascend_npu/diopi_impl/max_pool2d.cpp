/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    BEGIN_CALL_ACL_OP(out, indices, input, kernelSize, stride, padding, dilation);
    acl_op::max_pool2d_with_indices_out(inputAt, kernelSizeAt, strideAt, paddingAt, dilationAt, ceilMode, outAt, indicesAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMaxPool2dBackward1(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                     diopiConstTensorHandle_t input, diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                     bool ceilMode, diopiConstTensorHandle_t indices) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, input, kernelSize, stride, padding, dilation, indices);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(gradInputAt);
    gradInputAt = acl_op::max_pool2d_with_indices_backward(gradOutputAt, inputAt, kernelSizeAt, strideAt, paddingAt, dilationAt, ceilMode, indicesAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
