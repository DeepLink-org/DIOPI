/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

extern "C" {

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    BEGIN_CALL_ACL_OP(out, input, weight, bias, stride, padding, dilation);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(at_out);
    at_out = acl_op::npu_conv2d(at_input, at_weight, at_bias, at_stride, at_padding, at_dilation, groups);
    END_CALL_ACL_OP();
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                        diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                        diopiSize_t *biasSizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradBias, gradOutput, input, weight, biasSizes, stride, padding, dilation);
    if (gradInput) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(at_gradInput);
    }
    if (gradWeight) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(at_gradWeight);
    }
    if (gradBias) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(at_gradBias);
    }
    std::tie(at_gradInput, at_gradWeight, at_gradBias) =  acl_op::npu_conv2d_backward(at_input, at_gradOutput, at_weight, at_stride, at_padding, at_dilation, groups, {gradInput==nullptr, gradWeight==nullptr, gradBias==nullptr});
    END_CALL_ACL_OP();
}

}  // extern "C"