// /**
//  * @file
//  * @author DeepLink
//  * @copyright  (c) 2024, DeepLink.
//  */

// #include "helper.hpp"
// #include "op_plugin/AclOpsInterface.h"

// namespace OP_IMPL_NS {

// diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
//                                   diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
//     int64_t reductionValue = static_cast<int64_t>(reduction);
//     diopiTensorHandle_t weightTensor = weight;
//     ascend::AscendTensor weightT(weight);
//     if (!weightT.defined()) {
//     }
//     ascend::makeOnesLike() BEGIN_CALL_ACL_OP(gradOutput, gradInput, input, target, weight);
//     if (!inputAt.defined() || inputAt.numel() == 0 || !gradInputAt.defined()) {
//         return diopiSuccess;
//     }
//     acl_op::nll_loss_backward(gradOutputAt, inputAt, targetAt, weightAt, reductionValue, ignoreIndex, gradInputAt);

//     END_CALL_ACL_OP();
// }

// }  // namespace OP_IMPL_NS
