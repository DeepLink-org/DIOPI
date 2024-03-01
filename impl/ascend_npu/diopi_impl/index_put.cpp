/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                           diopiConstTensorHandle_t* indices, int64_t indicesCounts, bool accumulate) {
    BEGIN_CALL_ACL_OP(out, input, values);
    c10::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(indicesCounts);
    for (int i = 0; i < indicesCounts; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }

    outAt.copy_(inputAt);
    op_api::index_put_(outAt, indicesAtList, valuesAt, accumulate);
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

    op_api::index_put_(inputAt, indicesAtList, valuesAt, accumulate);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
