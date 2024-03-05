/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalizedShape,
                            double eps) {
    BEGIN_CALL_ACL_OP(out, saveMean, saveInvstd, input, weight, bias);
    std::vector<int64_t> normalizedVec(normalizedShape.data, normalizedShape.data + normalizedShape.len);
    auto result = op_api::native_layer_norm(inputAt, normalizedVec, weightAt, biasAt, eps);
    outAt.copy_(std::get<0>(result));
    saveMeanAt.copy_(std::get<1>(result));
    saveInvstdAt.copy_(std::get<2>(result));
    END_CALL_ACL_OP();
}

diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalizedShape) {
    BEGIN_CALL_ACL_OP(gradInput, gradWeight, gradBias, gradOutput, input, weight, bias, mean, rstd);
    std::vector<int64_t> normalizedVec(normalizedShape.data, normalizedShape.data + normalizedShape.len);
    std::array<bool, 3> maskVec{true, true, true};
    auto result = op_api::native_layer_norm_backward(gradOutputAt, inputAt, normalizedVec, meanAt, rstdAt, weightAt, biasAt, maskVec);
    if (gradInputAt.defined()) {
        gradInputAt.copy_(std::get<0>(result));
    } else {
        impl::aten::buildDiopiTensor(ctx, std::get<0>(result), &gradInput);
    }
    if (gradWeightAt.defined()) {
        gradWeightAt.copy_(std::get<1>(result));
    } else {
        impl::aten::buildDiopiTensor(ctx, std::get<1>(result), &gradWeight);
    }
    if (gradBiasAt.defined()) {
        gradBiasAt.copy_(std::get<2>(result));
    } else {
        impl::aten::buildDiopiTensor(ctx, std::get<2>(result), &gradBias);
    }
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
