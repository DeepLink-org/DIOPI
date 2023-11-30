/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

//namespace OP_IMPL_NS {
extern "C" {

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t runningMean,
                            diopiTensorHandle_t runningVar, bool training, double momentum, double eps) {
    BEGIN_CALL_ACL_OP(out, saveMean, saveInvstd, input, weight, bias, runningMean, runningVar);
    if (inputAt.dim() > 5) {
        at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    }
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(runningMeanAt);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(runningVarAt);
    acl_op::native_batch_norm_out(inputAt, weightAt, biasAt, runningMeanAt, runningVarAt, training, momentum, eps, outAt, saveMeanAt, saveInvstdAt);
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

}  // OP_IMPL_NS
