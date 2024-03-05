/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

namespace {
int64_t getReductionValue(const diopiReduction_t reduction) {
    int64_t reductionValue = 0;
    if (diopiReduction_t::ReductionNone == reduction) {
        reductionValue = 0;
    } else if (diopiReduction_t::ReductionMean == reduction) {
        reductionValue = 1;
    } else if (diopiReduction_t::ReductionSum == reduction) {
        reductionValue = 2;
    } else if (diopiReduction_t::ReductionEND == reduction) {
        reductionValue = 3;
    }
    return reductionValue;
}
}  // namespace

diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiReduction_t reduction) {
    BEGIN_CALL_ACL_OP(input, target, out);
    int64_t reductionValue = getReductionValue(reduction);
    op_api::mse_loss_out(inputAt, targetAt, reductionValue, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput, input, target);
    int64_t reductionValue = getReductionValue(reduction);
    op_api::mse_loss_backward_out(gradOutputAt, inputAt, targetAt, reductionValue, gradInputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
