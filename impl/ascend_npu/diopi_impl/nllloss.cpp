/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cmath>

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiNLLLossV1(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t totalWeight, diopiConstTensorHandle_t input,
                            diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    BEGIN_CALL_ACL_OP(out, input, target, weight, totalWeight);
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

diopiError_t diopiNLLLossV1Backward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t totalWeight, diopiReduction_t reduction, int64_t ignoreIndex) {
    BEGIN_CALL_ACL_OP(input, target, weight, gradInput, gradOutput, totalWeight);
    if (input == nullptr || gradInput == nullptr || inputAt.numel() <= 0 || gradInputAt.numel() <= 0) {
        return diopiSuccess;
    }
    /*
     * A tensor representing the sum of weights for each element considered in the NLL loss computation.
     * In case a weight tensor is provided, total_weight represents the sum of weights for all the non-ignored indices in the target tensor.
     * When no weight tensor is provided, total_weight corresponds to the count of all non-ignored indices.
     */

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
