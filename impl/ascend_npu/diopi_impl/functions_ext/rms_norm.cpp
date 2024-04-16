/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <c10/core/ScalarType.h>

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRms, diopiConstTensorHandle_t input,
                          diopiSize_t normalizedShape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    BEGIN_CALL_ACL_OP(out, invRms, input, weight);
    TORCH_CHECK(1 == normalizedShape.len && normalizedShape.data[0] == inputAt.size(inputAt.dim() - 1),
                "normalized shape currently must be the size of the last dimension on ascend!");
    if (inputAt.scalar_type() == at::kHalf || inputAt.scalar_type() == at::kBFloat16) {
        TORCH_CHECK(invRmsAt.scalar_type() == at::kFloat, "When the dtype of input is float16 or bfloat16, the dtype of invRms must be float32!");
    }

    if (false) {
        std::tuple<at::Tensor, at::Tensor> result;
        result = acl_op::npu_rms_norm(inputAt, weightAt, eps);
        invRmsAt.copy_(std::get<1>(result));
        outAt.copy_(std::get<0>(result));
    } else {
        EXEC_NPU_CMD(aclnnRmsNorm, inputAt, weightAt, eps, outAt, invRmsAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                  diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRms, diopiSize_t normalizedShape, double eps) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradOutput, input, weight, invRms);
    TORCH_CHECK(1 == normalizedShape.len && normalizedShape.data[0] == inputAt.size(inputAt.dim() - 1),
                "normalized shape currently must be the size of the last dimension on ascend!");
    if (inputAt.scalar_type() == at::kHalf || inputAt.scalar_type() == at::kBFloat16) {
        TORCH_CHECK(invRmsAt.scalar_type() == at::kFloat, "When the dtype of input is float16 or bfloat16, the dtype of invRms must be float32!");
    }

    if (false) {
        std::tuple<at::Tensor, at::Tensor> result;
        result = acl_op::npu_rms_norm_backward(gradOutputAt, inputAt, weightAt, invRmsAt);
        gradInputAt.copy_(std::get<0>(result));
        gradWeightAt.copy_(std::get<1>(result));
    } else {
        if (gradWeightAt.scalar_type() != at::kFloat) {
            at::Tensor gradWeightTempAt = at_npu::native::OpPreparation::apply_tensor_with_format(
                op_infer::rms_norm_grad_npu_output_size(inputAt, weightAt)[1], gradWeightAt.options().dtype(at::kFloat), ACL_FORMAT_ND);
            EXEC_NPU_CMD(aclnnRmsNormGrad, gradOutputAt, inputAt, invRmsAt, weightAt, gradInputAt, gradWeightTempAt);
            gradWeightAt.copy_(gradWeightTempAt);
        } else {
            EXEC_NPU_CMD(aclnnRmsNormGrad, gradOutputAt, inputAt, invRmsAt, weightAt, gradInputAt, gradWeightAt);
        }
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
