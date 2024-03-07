/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

static c10::List<c10::optional<at::Tensor>> castIntIndicesToLongIndices(const c10::List<c10::optional<at::Tensor>>& indices) {
    c10::List<c10::optional<at::Tensor>> result;
    for (c10::optional<at::Tensor> indexOpt : indices) {
        if (!indexOpt.has_value()) {
            result.emplace_back();
        } else {
            at::Tensor index = std::move(*indexOpt);
            result.emplace_back(index.scalar_type() == at::kInt ? index.toType(at::kLong) : index);
        }
    }
    return result;
}

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
    BEGIN_CALL_ACL_OP(input);
    torch::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(nums);
    for (int i = 0; i < nums; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }

    auto indicesCast = castIntIndicesToLongIndices(indicesAtList);
    at::Tensor outAt = op_api::index(inputAt, indicesCast);
    impl::aten::buildDiopiTensor(ctx, outAt, out);
    END_CALL_ACL_OP();
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t zerosLikeInput, diopiConstTensorHandle_t* indices,
                                int64_t nums, diopiConstTensorHandle_t gradOutput) {
    BEGIN_CALL_ACL_OP(gradInput, zerosLikeInput, gradOutput);
    torch::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(nums);
    for (int i = 0; i < nums; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }

    auto indicesCast = castIntIndicesToLongIndices(indicesAtList);
    op_api::_index_put_impl_(zerosLikeInputAt, indicesCast, gradOutputAt, true, false);
    gradInputAt.copy_(zerosLikeInputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
