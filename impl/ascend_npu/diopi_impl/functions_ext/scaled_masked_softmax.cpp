/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiScaledMaskedSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                      double scale, bool fixedTriuMask) {
    BEGIN_CALL_ACL_OP(input, mask, out);
    at_npu::native::OpCommand cmd;
    cmd.Name("ScaledMaskedSoftmax")
        .Input(inputAt)
        .Input(maskAt)
        .Output(outAt)
        .Attr<c10::Scalar>("scale", scale)
        .Attr<bool>("fixed_triu_mask", fixedTriuMask)
        .Run();
    END_CALL_ACL_OP();
}

diopiError_t diopiScaledMaskedSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                              diopiConstTensorHandle_t out, diopiConstTensorHandle_t mask, double scale, bool fixedTriuMask) {
    BEGIN_CALL_ACL_OP(gradOutput, out, mask, gradInput);
    at_npu::native::OpCommand cmd;
    cmd.Name("ScaledMaskedSoftmaxGrad")
        .Input(gradOutputAt)
        .Input(outAt)
        .Input(maskAt)
        .Output(gradInputAt)
        .Attr<c10::Scalar>("scale", scale)
        .Attr<bool>("fixed_triu_mask", fixedTriuMask)
        .Run();
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
