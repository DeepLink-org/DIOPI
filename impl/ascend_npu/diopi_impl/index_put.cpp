/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                           diopiConstTensorHandle_t* indices, int64_t indicesCounts, bool accumulate) {
    BEGIN_CALL_ACL_OP(out, input, values);
    // handle undefined tensor and empty tensor
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    outAt.copy_(inputAt);
    c10::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(indicesCounts);
    for (int i = 0; i < indicesCounts; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }
    acl_op::_index_put_impl_(outAt, indicesAtList, valuesAt, accumulate, false);
    END_CALL_ACL_OP();
}

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                              int64_t indicesCounts, bool accumulate) {
    BEGIN_CALL_ACL_OP(input, values);
    // handle undefined tensor and empty tensor
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    c10::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(indicesCounts);
    for (int i = 0; i < indicesCounts; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }
    acl_op::_index_put_impl_(inputAt, indicesAtList, valuesAt, accumulate, false);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
