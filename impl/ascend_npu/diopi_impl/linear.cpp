/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace {

at::Tensor transTensorTo2D(const at::Tensor& tensor) {
    std::vector<int64_t> dims;
    std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
    int64_t product = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<>());
    dims = {product, shape.back()};
    return impl::aten::viewStorage(tensor, dims);
}
}  // namespace

namespace OP_IMPL_NS {

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                         diopiConstTensorHandle_t bias) {
    BEGIN_CALL_ACL_OP(input, weight, bias, out);

    at::Tensor inputAt2D = inputAt;
    at::Tensor outAt2D = outAt;
    if (inputAt.dim() > 2) {
        inputAt2D = transTensorTo2D(inputAt);
    }
    if (outAt.dim() > 2) {
        outAt2D = transTensorTo2D(outAt);
    }

    at::Tensor weightAtT = weightAt.t();
    int8_t cubeMathType = at_npu::native::OpPreparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    if (biasAt.defined()) {
        const at::Scalar beta = 1;
        const at::Scalar alpha = 1;
        EXEC_NPU_CMD(aclnnAddmm, biasAt, inputAt2D, weightAtT, beta, alpha, outAt2D, cubeMathType);
    } else {
        EXEC_NPU_CMD(aclnnMm, inputAt2D, weightAtT, outAt2D, cubeMathType);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                 diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    BEGIN_CALL_ACL_OP(input, weight, gradOutput, gradInput, gradWeight, gradBias);
    at::Tensor gradOutputAt2D = gradOutputAt;
    if (gradOutputAt.dim() > 2) {
        gradOutputAt2D = transTensorTo2D(gradOutputAt);
    }

    at::Tensor gradInputAt2D = gradInputAt;
    if (gradInputAt.dim() > 2) {
        gradInputAt2D = transTensorTo2D(gradInputAt);
    }

    at::Tensor inputAt2D = inputAt;
    if (inputAt.dim() > 2) {
        inputAt2D = transTensorTo2D(inputAt);
    }

    int8_t cubeMathType = at_npu::native::OpPreparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnMm, gradOutputAt2D, weightAt, gradInputAt2D, cubeMathType);

    at::Tensor gradOutputAt2DT = gradOutputAt2D.t();
    EXEC_NPU_CMD(aclnnMm, gradOutputAt2DT, inputAt2D, gradWeightAt, cubeMathType);

    if (gradBiasAt.defined()) {
        auto outDim = gradOutputAt.dim();
        auto biasDim = gradBiasAt.dim();
        if (outDim > biasDim) {
            std::vector<int64_t> sumDims(outDim - biasDim);
            std::iota(sumDims.begin(), sumDims.end(), 0);
            op_api::sum_out(gradOutputAt, sumDims, false, gradBiasAt.scalar_type(), gradBiasAt);
        } else {
            gradBiasAt.copy_(gradOutputAt);
        }
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
