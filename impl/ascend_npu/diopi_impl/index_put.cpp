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

diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                           diopiConstTensorHandle_t* indices, int64_t indicesCounts, bool accumulate) {
    BEGIN_CALL_ACL_OP(out, input, values);
    c10::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(indicesCounts);
    for (int i = 0; i < indicesCounts; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }

    outAt.copy_(inputAt);
    auto indicesCast = castIntIndicesToLongIndices(indicesAtList);
    op_api::_index_put_impl_(outAt, indicesCast, valuesAt, accumulate, false);
    END_CALL_ACL_OP();
}

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                              int64_t indicesCounts, bool accumulate) {
    BEGIN_CALL_ACL_OP(input, values);
    c10::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(indicesCounts);
    for (int i = 0; i < indicesCounts; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }

    auto indicesCast = castIntIndicesToLongIndices(indicesAtList);
    op_api::_index_put_impl_(inputAt, indicesCast, valuesAt, accumulate, false);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
