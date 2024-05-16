/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

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

namespace impl {
namespace ascend {

diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiReduction_t reduction) {
    int64_t reductionValue = getReductionValue(reduction);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMseLoss, ctx, input, target, reduction, out);
    return diopiSuccess;
}

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    int64_t reductionValue = getReductionValue(reduction);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMseLossBackward, ctx, gradOutput, input, target, reductionValue, gradInput);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
