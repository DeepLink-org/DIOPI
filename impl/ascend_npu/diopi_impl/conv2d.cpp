/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

//namespace OP_IMPL_NS {
extern "C" {

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    BEGIN_CALL_ACL_OP(out, input, weight, bias, stride, padding, dilation);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    outAt = acl_op::npu_conv2d(inputAt, weightAt, biasAt, strideAt, paddingAt, dilationAt, groups);
    END_CALL_ACL_OP();
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                        diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                        diopiSize_t *biasSizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradBias, gradOutput, input, weight, biasSizes, stride, padding, dilation);
    if (gradInput) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(gradInputAt);
    }
    if (gradWeight) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(gradWeightAt);
    }
    if (gradBias) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(gradBiasAt);
    }
    std::tie(gradInputAt, gradWeightAt, gradBiasAt) =  acl_op::npu_conv2d_backward(inputAt, gradOutputAt, weightAt, strideAt, paddingAt, dilationAt, groups, {gradInput==nullptr, gradWeight==nullptr, gradBias==nullptr});
    END_CALL_ACL_OP();
}

}  // OP_IMPL_NS