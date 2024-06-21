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

    EXEC_NPU_CMD(aclnnRmsNorm, inputAt, weightAt, eps, outAt, invRmsAt);
    if (bias) {
        auto biasAt = impl::aten::buildATen(bias);
        op_api::add_(outAt, biasAt, 1.0);
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

    if (gradWeightAt.scalar_type() != at::kFloat) {
        at::Tensor gradWeightTempAt = at_npu::native::OpPreparation::apply_tensor_with_format(
            op_infer::rms_norm_grad_npu_output_size(inputAt, weightAt)[1], gradWeightAt.options().dtype(at::kFloat), ACL_FORMAT_ND);
        EXEC_NPU_CMD(aclnnRmsNormGrad, gradOutputAt, inputAt, invRmsAt, weightAt, gradInputAt, gradWeightTempAt);
        gradWeightAt.copy_(gradWeightTempAt, true);
    } else {
        EXEC_NPU_CMD(aclnnRmsNormGrad, gradOutputAt, inputAt, invRmsAt, weightAt, gradInputAt, gradWeightAt);
    }
    if (gradBias) {
        auto gradBiasAt = impl::aten::buildATen(gradBias);
        auto outDim = gradOutputAt.dim();
        auto biasDim = gradBiasAt.dim();
        if (outDim > biasDim) {
            std::vector<int64_t> sumDims(outDim - biasDim);
            std::iota(sumDims.begin(), sumDims.end(), 0);
            op_api::sum_out(gradOutputAt, sumDims, false, gradBiasAt.scalar_type(), gradBiasAt);
        } else {
            gradBiasAt.copy_(gradOutputAt, true);
        }
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
