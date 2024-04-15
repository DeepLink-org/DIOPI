/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                         diopiConstTensorHandle_t bias) {
    BEGIN_CALL_ACL_OP(input, weight, out);
    const at::Tensor& weightT = weightAt.t();
    int8_t cubeMathType = at_npu::native::OpPreparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    if (bias) {
        std::cout << "非空" << std::endl;
        std::cout << "check bias: " << bias << std::endl;
        const at::Scalar beta = 1;
        const at::Scalar alpha = 1;
        at::Tensor biasAt = impl::aten::buildATen(bias);
        EXEC_NPU_CMD(aclnnAddmm, biasAt, inputAt, weightT, beta, alpha, outAt, cubeMathType);
    } else {
        std::cout << "空" << std::endl;
        std::cout << "check bias: " << bias << std::endl;
        EXEC_NPU_CMD(aclnnMm, input, weightT, outAt, cubeMathType);
    }
    END_CALL_ACL_OP();
}

// diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
//                                  diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
//     BEGIN_CALL_ACL_OP(input, weight, gradOutput, gradInput, gradWeight, gradBias);
//     int8_t cubeMathType = at_npu::native::OpPreparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
//     if (gradBias.defined()) {
//     } else {
//         EXEC_NPU_CMD(aclnnMm, gradOutputAt, weight, gradInputAt, cubeMathType);
//     }

//     const at::Tensor& grad_t = grad.t();
//     at::Tensor weight_grad = npu_preparation::apply_tensor_without_format(weight.sizes(), grad.options());
//     EXEC_NPU_CMD(aclnnMm, grad_t, input, weight_grad, cube_math_type);

//     return std::tie(input_grad, weight_grad);
// }

}  // namespace OP_IMPL_NS
