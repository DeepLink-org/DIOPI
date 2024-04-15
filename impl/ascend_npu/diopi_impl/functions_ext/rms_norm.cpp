/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRms, diopiConstTensorHandle_t input,
                          diopiSize_t normalizedShape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    BEGIN_CALL_ACL_OP(out, invRms, input, weight);
    TORCH_CHECK(
        1 == normalizedShape.len && normalizedShape.data[0] == inputAt.size(inputAt.dim() - 1) || normalizedShape.len == 0 || normalizedShape.data == nullptr,
        "normalized shape is not supported on ascend!");

    std::tuple<at::Tensor, at::Tensor> result;
    if (false) {
        result = acl_op::npu_rms_norm(inputAt, weightAt, eps);
        invRmsAt.copy_(std::get<1>(result));
        outAt.copy_(std::get<0>(result));
    } else {
        if (invRmsAt.scalar_type() != at::kFloat) {
            at::Tensor invRmsTempAt = at_npu::native::OpPreparation::apply_tensor_with_format(
                op_infer::rms_norm_npu_output_size(inputAt, weightAt)[1], invRmsAt.options().dtype(at::kFloat), ACL_FORMAT_ND);
            EXEC_NPU_CMD(aclnnRmsNorm, inputAt, weightAt, eps, outAt, invRmsTempAt);
            invRmsAt.copy_(invRmsTempAt);
        } else {
            EXEC_NPU_CMD(aclnnRmsNorm, inputAt, weightAt, eps, outAt, invRmsAt);
        }
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                  diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRms, diopiSize_t normalizedShape, double eps) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradOutput, input, weight, invRms);
    TORCH_CHECK(
        1 == normalizedShape.len && normalizedShape.data[0] == inputAt.size(inputAt.dim() - 1) || normalizedShape.len == 0 || normalizedShape.data == nullptr,
        "normalized shape is not supported on ascend!");
    at::Tensor invRmsTempAt = invRmsAt;
    if (invRmsAt.scalar_type() != at::kFloat) {
        invRmsTempAt = invRmsAt.to(at::kFloat);
    }

    if (false) {
        std::tuple<at::Tensor, at::Tensor> result;
        result = acl_op::npu_rms_norm_backward(gradOutputAt, inputAt, weightAt, invRmsTempAt);
        gradInputAt.copy_(std::get<0>(result));
        gradWeightAt.copy_(std::get<1>(result));
    } else {
        if (gradWeightAt.scalar_type() != at::kFloat) {
            at::Tensor gradWeightTempAt = at_npu::native::OpPreparation::apply_tensor_with_format(
                op_infer::rms_norm_grad_npu_output_size(inputAt, weightAt)[1], gradWeightAt.options().dtype(at::kFloat), ACL_FORMAT_ND);
            EXEC_NPU_CMD(aclnnRmsNormGrad, gradOutputAt, inputAt, invRmsTempAt, weightAt, gradInputAt, gradWeightTempAt);
            gradWeightAt.copy_(gradWeightTempAt);
        } else {
            EXEC_NPU_CMD(aclnnRmsNormGrad, gradOutputAt, inputAt, invRmsTempAt, weightAt, gradInputAt, gradWeightAt);
        }
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
