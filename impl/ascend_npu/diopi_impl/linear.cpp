/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/utils/op_api_common.h"

namespace {

at::Tensor transTensorTo2D(const at::Tensor& tensor) {
    std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
    int64_t product = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<>());
    std::vector<int64_t> dims = {product, shape.back()};
    return impl::aten::viewStorage(tensor, dims);
}
}  // namespace

namespace OP_IMPL_NS {

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                         diopiConstTensorHandle_t bias) {
    BEGIN_CALL_ACL_OP(input, weight, bias, out);

    at::Tensor inputAt2D = (inputAt.dim() > 2) ? transTensorTo2D(inputAt) : inputAt;
    at::Tensor outAt2D = (outAt.dim() > 2) ? transTensorTo2D(outAt) : outAt;
    at::Tensor weightAt2D = (weightAt.dim() > 2) ? transTensorTo2D(weightAt) : weightAt;
    at::Tensor weightAt2DT = weightAt2D.t();

    int8_t cubeMathType = at_npu::native::OpPreparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnMatmul, inputAt2D, weightAt2DT, outAt2D, cubeMathType);

    if (biasAt.defined()) {
        at::Scalar alpha = 1;
        EXEC_NPU_CMD(aclnnInplaceAdd, outAt, biasAt, alpha);
    }

    END_CALL_ACL_OP();
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                 diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    BEGIN_CALL_ACL_OP(input, weight, gradOutput, gradInput, gradWeight, gradBias);

    int8_t cubeMathType = at_npu::native::OpPreparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnMatmul, gradOutputAt, weightAt, gradInputAt, cubeMathType);

    at::Tensor inputAt2D = (inputAt.dim() > 2) ? transTensorTo2D(inputAt) : inputAt;
    at::Tensor gradOutputAt2D = (gradOutputAt.dim() > 2) ? transTensorTo2D(gradOutputAt) : gradOutputAt;
    at::Tensor gradWeightAt2D = (gradWeightAt.dim() > 2) ? transTensorTo2D(gradWeightAt) : gradWeightAt;
    at::Tensor gradOutputAt2DT = gradOutputAt2D.t();
    EXEC_NPU_CMD(aclnnMatmul, gradOutputAt2DT, inputAt2D, gradWeightAt2D, cubeMathType);

    if (gradBiasAt.defined()) {
        auto outDim = gradOutputAt.dim();
        auto biasDim = gradBiasAt.dim();
        std::vector<int64_t> sumDims(outDim - biasDim);
        std::iota(sumDims.begin(), sumDims.end(), 0);
        bool keepDim = false;
        auto dtype = gradBiasAt.scalar_type();
        at::IntArrayRef sumDimsArrayRef(sumDims);
        EXEC_NPU_CMD(aclnnReduceSum, gradOutputAt, sumDimsArrayRef, keepDim, dtype, gradBiasAt);
    }

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
