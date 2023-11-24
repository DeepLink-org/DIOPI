/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

extern "C" {

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t runningMean,
                            diopiTensorHandle_t runningVar, bool training, double momentum, double eps) {
    BEGIN_CALL_ACL_OP(out, saveMean, saveInvstd, input, weight, bias, runningMean, runningVar);
    if (at_input.dim() > 5) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(at_out);
    }
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(at_out);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(at_runningMean);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(at_runningVar);
    //::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out(const at::Tensor & input, const c10::optional<at::Tensor> & weight,
    // const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training,
    // double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd);
    acl_op::native_batch_norm_out(at_input, at_weight, at_bias, at_runningMean, at_runningVar, training, momentum, eps, at_out, at_saveMean, at_saveInvstd);
    END_CALL_ACL_OP();
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t runningMean, diopiConstTensorHandle_t runningVar, diopiConstTensorHandle_t saveMean,
                                    diopiConstTensorHandle_t saveInvstd, bool training, double eps) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradBias, gradOutput,  input, weight, runningMean, runningVar, saveMean, saveInvstd);

    //::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const
    //:c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const
    //:c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask);
    END_CALL_ACL_OP();
}

}  // extern "C"
