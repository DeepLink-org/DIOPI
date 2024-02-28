/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cmath>

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

//: std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight,
//: int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight); :std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out(const
//: at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output,
//: at::Tensor & total_weight);
diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    BEGIN_CALL_ACL_OP(out, input, target, weight);
    if (input == nullptr) {
        return diopiSuccess;
    }
    if (inputAt.numel() <= 0) {
        if (diopiReduction_t::ReductionMean == reduction) {
            op_api::fill_(outAt, c10::Scalar(std::nanf("")));
        } else if (diopiReduction_t::ReductionSum == reduction || diopiReduction_t::ReductionNone == reduction) {
            op_api::fill_(outAt, c10::Scalar(0.0f));
        }
        return diopiSuccess;
    }

    at::Tensor totalWeightAt = at_npu::native::empty_npu({1}, inputAt.options());
    const int64_t C = inputAt.dim() >= 4 ? inputAt.size(1) : inputAt.size(-1);
    if (weight == nullptr) {
        weightAt = at_npu::native::empty_npu({C}, inputAt.options());
        op_api::fill_(weightAt, c10::Scalar(1.0f));
    }
    if (inputAt.dim() <= 2) {
        op_api::nll_loss_forward_out(inputAt, targetAt, weightAt, reduction, ignoreIndex, outAt, totalWeightAt);
    } else if (inputAt.dim() == 4) {
        op_api::nll_loss2d_forward_out(inputAt, targetAt, weightAt, reduction, ignoreIndex, outAt, totalWeightAt);
    } else {
        auto veiwedInputAt = impl::aten::viewStorage(inputAt, {inputAt.size(0), inputAt.size(1), inputAt.numel() / inputAt.size(0) / inputAt.size(1), 1});
        auto veiwedOutAt = (outAt.numel() > 1) ? impl::aten::viewStorage(outAt, {outAt.size(0), outAt.numel() / outAt.size(0), 1}) : outAt;
        auto veiwedTargetAt = impl::aten::viewStorage(targetAt, {targetAt.size(0), targetAt.numel() / targetAt.size(0), 1});
        op_api::nll_loss2d_forward_out(veiwedInputAt, veiwedTargetAt, weightAt, reduction, ignoreIndex, veiwedOutAt, totalWeightAt);
    }
    END_CALL_ACL_OP();
}
#if 0
// at::Tensor nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight);
// at::Tensor nll_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight);
diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    BEGIN_CALL_ACL_OP(input, target, weight, gradInput, gradOutput);
    if (input == nullptr) {
        return diopiSuccess;
    }
    at::Tensor totalWeightAt;
    if (weight != nullptr) {
        totalWeightAt = at_npu::native::empty_npu(weightAt.sizes(), weightAt.options());
    }

    if (inputAt.dim() <= 2) {
        op_api::nll_loss_backward(gradOutputAt, inputAt, targetAt, weightAt, reduction, ignoreIndex, totalWeightAt);
    } else if (inputAt.dim() == 4) {
        op_api::nll_loss2d_backward(gradOutputAt, inputAt, targetAt, weightAt, reduction, ignoreIndex,
        totalWeightAt);
    } else {
        TORCH_CHECK(false, "diopiNLLLossBackward: invalid input dim");
    }
    END_CALL_ACL_OP();
}
#endif

}  // namespace OP_IMPL_NS
