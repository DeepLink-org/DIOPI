/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace {

at::Tensor& conv2d_backward_bias_out_nocheck(at::Tensor& grad_bias, const at::Tensor& grad) {
    if (grad.numel() == grad.size(0) * grad.size(1)) {
        // at::Tensor grad_view = grad.contiguous().view({grad.size(0), grad.size(1)});
        at::Tensor grad_view = impl::aten::view(grad, {grad.size(0), grad.size(1)});
        acl_op::sum_out(grad_view, c10::SmallVector<int64_t, N>{0}, false, grad_view.scalar_type(), grad_bias);
    } else {
        // at::Tensor grad_view = grad.contiguous().view({grad.size(0), grad.size(1), -1});
        at::Tensor grad_view = impl::aten::view(grad, {grad.size(0), grad.size(1), grad.numel() / grad.size(0) / grad.size(1)});
        acl_op::sum_out(grad_view, c10::SmallVector<int64_t, N>{0, 2}, false, grad_view.scalar_type(), grad_bias);
    }

    return grad_bias;
}

}  // namespace

// namespace OP_IMPL_NS {
extern "C" {

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    BEGIN_CALL_ACL_OP(out, input, weight, bias, stride, padding, dilation);
    if (c10::multiply_integers(inputAt.sizes()) <= 0) {
        return diopiSuccess;
    }
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    outAt = acl_op::npu_conv2d(inputAt, weightAt, biasAt, strideAt, paddingAt, dilationAt, groups);
    END_CALL_ACL_OP();
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                        diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                        diopiSize_t* biasSizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradBias, gradOutput, input, weight, biasSizes, stride, padding, dilation);
    if (c10::multiply_integers(inputAt.sizes()) <= 0) {
        return diopiSuccess;
    }
    if (gradInput) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(gradInputAt);
    }
    if (gradWeight) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(gradWeightAt);
    }

    acl_op::npu_conv2d_backward(
        inputAt, gradOutputAt, weightAt, strideAt, paddingAt, dilationAt, groups, {gradInput != nullptr, gradWeight != nullptr, false});
    if (gradBias != nullptr) {
        conv2d_backward_bias_out_nocheck(gradBiasAt, gradOutputAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
