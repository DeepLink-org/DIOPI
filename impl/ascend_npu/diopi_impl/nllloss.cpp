/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cmath>

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

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
    const int64_t channel = inputAt.dim() >= 4 ? inputAt.size(1) : inputAt.size(-1);
    if (weight == nullptr) {
        weightAt = at_npu::native::empty_npu({channel}, inputAt.options());
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

inline at::Tensor calTotalWeight(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& target, const int64_t ignoreIndex) {
    at::Tensor totalWeight;
    // Flatten the target tensor for easier processing
    auto flatTarget = target.view(-1);

    // calculate totalWeight
    // Create a mask corresponding to ignoreIndex if it's provided
    auto mask = at_npu::native::empty_npu(flatTarget.sizes(), flatTarget.options().dtype(at::kBool));
    if (ignoreIndex >= 0) {
        op_api::ne_out(flatTarget, at::Scalar(ignoreIndex), mask);
    } else {
        mask.fill_(true);
    }
    if (weight.defined()) {
        // Filter out the targets using the mask and compute total weight using index_select
        auto selectedTargetAt = op_api::masked_select(flatTarget, mask);
        auto selectedWeightAt = op_api::index_select(weight, 0, selectedTargetAt);
        totalWeight = op_api::sum(selectedWeightAt, weight.scalar_type());
    } else {
        // If weight is not defined, compute total weight by counting the valid targets
        totalWeight = op_api::sum(mask, input.scalar_type());
    }
    return totalWeight;
}

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    BEGIN_CALL_ACL_OP(input, target, weight, gradInput, gradOutput);
    if (input == nullptr || gradInput == nullptr || inputAt.numel() <= 0 || gradInputAt.numel() <= 0) {
        return diopiSuccess;
    }
    /*
     * A tensor representing the sum of weights for each element considered in the NLL loss computation.
     * In case a weight tensor is provided, total_weight represents the sum of weights for all the non-ignored indices in the target tensor.
     * When no weight tensor is provided, total_weight corresponds to the count of all non-ignored indices.
     */
    at::Tensor totalWeightAt = calTotalWeight(inputAt, weightAt, targetAt, ignoreIndex);

    if (inputAt.dim() <= 2) {
        op_api::nll_loss_backward_out(gradOutputAt, inputAt, targetAt, weightAt, reduction, ignoreIndex, totalWeightAt, gradInputAt);
    } else if (inputAt.dim() == 4) {
        op_api::nll_loss2d_backward_out(gradOutputAt, inputAt, targetAt, weightAt, reduction, ignoreIndex, totalWeightAt, gradInputAt);
    } else {
        auto veiwedInputAt = impl::aten::viewStorage(inputAt, {inputAt.size(0), inputAt.size(1), inputAt.numel() / inputAt.size(0) / inputAt.size(1), 1});
        auto veiwedGradInputAt = impl::aten::viewStorage(
            gradInputAt, {gradInputAt.size(0), gradInputAt.size(1), gradInputAt.numel() / gradInputAt.size(0) / gradInputAt.size(1), 1});
        auto veiwedGradOutAt = (gradOutputAt.numel() > 1)
                                   ? impl::aten::viewStorage(gradOutputAt, {gradOutputAt.size(0), gradOutputAt.numel() / gradOutputAt.size(0), 1})
                                   : gradOutputAt;
        auto veiwedTargetAt = impl::aten::viewStorage(targetAt, {targetAt.size(0), targetAt.numel() / targetAt.size(0), 1});
        op_api::nll_loss2d_backward_out(veiwedGradOutAt, veiwedInputAt, veiwedTargetAt, weightAt, reduction, ignoreIndex, totalWeightAt, veiwedGradInputAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
